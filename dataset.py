from tensorflow.keras.utils import Sequence
import numpy as np
import tensorflow as tf

def padding_mask(decoder_input, pad_seq_id=1): 
    mask = tf.math.equal(decoder_input, pad_seq_id)
    #convert 0 -> False, 1 -> True
    mask = tf.cast(mask, tf.int64)
    # add 1 dimension for batch
    mask = tf.expand_dims(mask, axis=0) # (1, seq_len)
    return mask

def look_ahead_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return tf.cast(mask, tf.int64) # (seq_len, seq_len)
class ImageCaptionDataset(Sequence):
    def __init__(self, df, X_col, y_col, batch_size, tokenizer, 
                 max_length, features,shuffle=True):
    
        self.df = df.copy() # Make a copy of the dataframe
        self.X_col = X_col # Column name containing the image names
        self.y_col = y_col # Column name containing the captions
        self.batch_size = batch_size # Batch size
        self.tokenizer = tokenizer # Tokenizer used for tokenizing captions
        self.max_length = max_length # Maximum length of caption (in tokens)
        self.features = features # Dictionary containing features of images
        self.shuffle = shuffle # Shuffle dataframe at the end of epoch
        self.n = len(self.df) # Length of dataframe
        
        self.pad_seq_id = tokenizer.token_to_id('[PAD]')
        self.end_seq_id = tokenizer.token_to_id('[EOS]')
        self.start_seq_id = tokenizer.token_to_id('[SOS]')
        
    def on_epoch_end(self): # Shuffle dataframe at the end of epoch 
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self): # Denotes the number of batches per epoch
        return self.n // self.batch_size
    
    def __getitem__(self,index): # Generate one batch of data
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size]
        dataloaded = self.__get_data(batch)
        return dataloaded
    
    def __get_data(self,batch): # Generate data containing batch_size samples
        encoder_output = []
        decoder_input = []
        decoder_mask = []
        label = []
        
        images = batch[self.X_col].tolist()
        for image in images:
            feature = self.features[image][0]
            sub_decoder_mask = []
              
            captions = batch.loc[batch[self.X_col]==image, self.y_col].tolist()
            for caption in captions:
                #encoder_output
                enc_out = tf.constant([feature], dtype=tf.float32)
                encoder_output.append(enc_out)
                #decoder input 
                dec_input = tf.constant(self.tokenizer.encode(caption).ids, dtype=tf.int64)
                dec_num_padding_tokens = self.max_length - dec_input.shape[0] - 1
                dec_pad_input = tf.constant([self.pad_seq_id]*dec_num_padding_tokens, dtype=tf.int64)
                dec_sos_input = tf.constant([self.start_seq_id], dtype=tf.int64)
                dec_input = tf.concat([dec_sos_input, dec_input, dec_pad_input], axis=0)
                dec_input = tf.expand_dims(dec_input, axis=0)
                #decoder mask 
                dec_padding_mask = padding_mask(dec_input, self.pad_seq_id)
                dec_lookahead_mask = look_ahead_mask(self.max_length)
                dec_mask = tf.maximum(dec_padding_mask, dec_lookahead_mask)
                #label
                lab = tf.constant(self.tokenizer.encode(caption).ids, dtype=tf.int64)
                lab_padding_tokens = dec_num_padding_tokens
                lab_eos = tf.constant([self.end_seq_id], dtype=tf.int64)
                lab_pad = tf.constant([self.pad_seq_id]*lab_padding_tokens, dtype=tf.int64)
                lab = tf.concat([lab, lab_eos, lab_pad], axis=0)
                lab = tf.expand_dims(lab, axis=0)
                
                
                
                decoder_input.append(dec_input)
                label.append(lab)
                sub_decoder_mask.append(dec_mask)
            
            sub_decoder_mask = tf.concat(sub_decoder_mask, axis=0)
            decoder_mask.append(sub_decoder_mask)
        
        encoder_output = tf.concat(encoder_output, axis=0)
        decoder_input = tf.concat(decoder_input, axis=0)
        decoder_mask = tf.concat(decoder_mask, axis=0)
        label = tf.concat(label, axis=0)
               
        return {
            'encoder_output': encoder_output,
            'decoder_input': decoder_input,
            'decoder_mask': decoder_mask,
            'label': label
        }
        