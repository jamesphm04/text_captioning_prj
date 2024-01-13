from decoder import build_decoder
from config import get_config, get_weights_file_path, latest_weights_file_path
from data_utils import get_ds

import tensorflow as tf
import numpy as np 

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy


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
        
    loss_fn = SparseCategoricalCrossentropy(ignore_class=tokenizer.token_to_id('[PAD]')) #TODO label_smoothing=0.1 ?
    
    @tf.function
    def train_step(decoder_input, encoder_output, decoder_mask, label):
        with tf.GradientTape() as tape:
            output = model(decoder_input, encoder_output, decoder_mask)
            loss = loss_fn(label, output, sample_weight=tf.cast(tf.math.not_equal(label, tokenizer.token_to_id('[PAD]')), tf.float32))
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradient(zip(gradients, model.trainable_variables))
        return loss
    
    for epoch in range(initial_epoch, config['num_epochs']):
        # model.reset_states()
        for batch in train_dataloader:
            decoder_input = batch['decoder_input']
            encoder_output = batch['encoder_output']
            decoder_mask = batch['decoder_mask']
            label = batch['label']
            
            print('label:', label)
            
            
# if __name__ == '__main__':
#     config = get_config()
#     train(config)