from tensorflow.keras.utils import Sequence
import numpy as np
import tensorflow as tf
 
class ImageCaptionGenerator(Sequence):
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
        
    def on_epoch_end(self): # Shuffle dataframe at the end of epoch 
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self): # Denotes the number of batches per epoch
        return self.n // self.batch_size
    
    def __getitem__(self,index): # Generate one batch of data
    
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size,:]
        dataloaded = self.__get_data(batch)        
        return dataloaded
    
    def __get_data(self,batch): # Generate data containing batch_size samples
        
        encoder_input = tf.zeros((1, 1920), dtype=tf.float32) # could be float32
        decoder_input = tf.zeros((1, self.max_length), dtype=tf.int64)
        
        images = batch[self.X_col].tolist()
           
        for image in images:
            feature = self.features[image][0]
            
            captions = batch.loc[batch[self.X_col]==image, self.y_col].tolist()
            for caption in captions:
                print(caption)
                enc_input = tf.constant([feature], dtype=tf.float32)
                dec_input = tf.constant(self.tokenizer.encode(caption).ids, dtype=tf.int64)
                
                dec_num_padding_tokens = self.max_length - dec_input.shape[0]
                dec_pad_input = tf.constant([self.pad_seq_id]*dec_num_padding_tokens, dtype=tf.int64)
                dec_input = tf.concat([dec_input, dec_pad_input], axis=0)
                
                dec_input = tf.expand_dims(dec_input, axis=0)
                
                print(f'decoder_input: {decoder_input.shape}, dec_input: {dec_input.shape}')
                
                encoder_input = tf.concat([encoder_input, enc_input], axis=0)
                decoder_input = tf.concat([decoder_input, dec_input], axis=0)
                
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
        }