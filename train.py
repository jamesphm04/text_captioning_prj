from decoder import build_decoder
from config import get_config, get_weights_file_path, latest_weights_file_path
from data_utils import get_ds
from tqdm import tqdm 

import tensorflow as tf
import numpy as np 

from dataset import look_ahead_mask

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


class CrossEntropyLoss(tf.keras.losses.Loss):
    def __init__(self, ignore_index, label_smoothing, **kwards):
        super().__init__(**kwards)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, self.ignore_index)
        num_class = y_pred.shape[-1]
        y_true_smoothed = tf.one_hot(y_true, num_class)
        y_true_smoothed = y_true_smoothed * (1 - self.label_smoothing) + self.label_smoothing / num_class
        
        loss = categorical_crossentropy(y_true_smoothed, y_pred)
        
        #apply the mask 
        loss = loss * tf.cast(mask, loss.dtype)
        
        #average the loss over non-masked positions
        loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(mask, loss.dtype))
        
        return loss

def train_step(decoder_input, encoder_output, decoder_mask, label, optimizer, loss_fn, model, vocab_size, pad_id=1):
    with tf.GradientTape() as tape:
        output = model(decoder_input, encoder_output, decoder_mask)
        flat_output = tf.reshape(output, (-1, vocab_size))
        flat_label = tf.reshape(label, (-1))
        
        
        loss = loss_fn(flat_label, flat_output)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(config):
    device = '/gpu:0' if tf.config.experimental.list_physical_devices('GPU') else '/cpu:0'
    
    #make sure the weight folder exist
    model_folder = f'{config["datasource"]}_{config["model_folder"]}'
    tf.io.gfile.makedirs(model_folder)
    
    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = build_decoder(tokenizer.get_vocab_size(), config['max_length'], config['d_model'], 6, config['num_heads'], config['dropout'], config['d_ff'])
    
    optimizer = Adam(learning_rate=config['lr'], epsilon=1e-09)
    
    # load the model if it exists
    initial_epoch = 0
    global_step = 0
    version = config['preload']
    model_filename = latest_weights_file_path(config) if version == 'latest' else get_weights_file_path(config, version) if version else None
    
    if model_filename:
        model.load_weights(model_filename)
        initial_epoch = int(model_filename.split('/')[-1].split('.')[0]) + 1
        global_step = initial_epoch * len(train_dataloader)
        print(f'Loaded weights from {model_filename}')
    else:
        print('No weights found')
        
    loss_fn = CrossEntropyLoss(ignore_index=tokenizer.token_to_id('[PAD]'), label_smoothing=0.1) #TODO label_smoothing=0.1 ?
    

    
    train_batch_iterator = tqdm(train_dataloader, desc='Training', dynamic_ncols=True)
    val_batch_iterator = tqdm(val_dataloader, desc='Validating', dynamic_ncols=True)
    
    
    for epoch in range(initial_epoch, config['num_epochs']):
        # model.reset_states() #TODO is this needed?      
        for batch in train_batch_iterator:
            decoder_input = batch['decoder_input']
            encoder_output = batch['encoder_output']
            decoder_mask = batch['decoder_mask']
            label = batch['label']
            loss = train_step(decoder_input, encoder_output, decoder_mask, label, optimizer, loss_fn, model, tokenizer.get_vocab_size())
            train_batch_iterator.set_postfix({'loss': f'{loss.numpy():6.3f}'})
            global_step += 1
            run_validation(model, val_batch_iterator, tokenizer, config['max_length'], lambda msg: train_batch_iterator.write(msg), global_step)
            
        if epoch % 1 == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            model.save(model_filename)
            
def run_validation(model, val_batch_iterator, tokenizer, max_length, print_msg, global_step, num_examples=2):
    model.trainable = False
    count = 0
    
    
    for batch in val_batch_iterator:
        count += 1
        
        encoder_output = batch['encoder_output']
        
        assert encoder_output.shape[0] == 1, 'Batch size must be 1 for validation'
        
        model_out = greedy_decode(model, encoder_output, tokenizer, max_length)
        
        src_img = batch['src_imgs'][0]
        label = batch['label'][0] # for visualization
        model_out_text = tokenizer.decode(model_out)
        tgt_caption = batch['tgt_captions'][0]
        
        #print the source, target and predicted captions
        print_msg('-' * 80)
        print_msg(f"{f'SOURCE: ':>12}{src_img}")
        print_msg(f"{f'TARGET: ':>12}{tgt_caption}")
        print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
        
        if count == num_examples:
            print_msg('-' * 80)
            break

def greedy_decode(model, encoder_output, tokenizer, max_length):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    
    #initialize the decoder input with sos_idx
    decoder_input = tf.constant([[sos_idx]], dtype=tf.int64)
    
    while True:
        if decoder_input.shape[1] == max_length:
            break
        decoder_mask = look_ahead_mask(decoder_input.shape[1])
        #calculate output
        out = model(decoder_input, encoder_output, tf.expand_dims(decoder_mask, axis=0))
        # get the token has the highest probability
        next_token = tf.argmax(out[:, -1], axis=-1)
        decoder_input = tf.concat([decoder_input, tf.expand_dims(next_token, axis=1)], axis=1)
        if next_token == eos_idx:
            break
    return decoder_input.numpy().squeeze()

if __name__ == '__main__':
    config = get_config()
    train(config)