import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np

class InputEmbeddings(layers.Layer):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = layers.Embedding(vocab_size, d_model)
        
    def call(self, x):
        return self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
class PositionalEncoding(layers.Layer):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = layers.Dropout(dropout)
        
        self.positional_encoding = self.positional_encoding()
        
    def positional_encoding(self):
        pe = np.zeros((self.seq_len, self.d_model))
        position = np.arange(0, self.seq_len, dtype=np.float32)[:, np.newaxis] #(seq_len, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32)*(-tf.math.log(10000.0)/self.d_model))
        pe[:, 0::2] = np.sin(position*div_term)
        pe[:, 1::2] = np.cos(position*div_term)
        pe = pe[np.newaxis, ...]
        self.pe = tf.cast(pe, tf.float32) #Q: is that savable in the model?
        
    def call(self, x):
        x = x + self.pe[:, tf.shape(x)[1], :]
        return self.dropout(x)
    
class LayerNormalization(layers.Layer):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = self.add_weight(name='alpha', shape=(d_model,), initializer='ones', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(d_model,), initializer='ones', trainable=True)
        
    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias
    
class FeedForwardBlock(layers.Layer):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear_1 = layers.Dense(d_ff, activation='relu')
        self.dropout = layers.Dropout(dropout)
        self.linear_2 = layers.Dense(d_model)
        
    def call(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_k = d_model // num_heads
        self.dropout = layers.Dropout(dropout)
        
        self.w_q = layers.Dense(d_model, use_bias=False)
        self.w_k = layers.Dense(d_model, use_bias=False)
        self.w_v = layers.Dense(d_model, use_bias=False)
        self.w_o = layers.Dense(d_model, use_bias=False)
        
        self.dropout = layers.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: layers.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = tf.matmul(query, key, transpose_b=True)
        
        if mask is not None:
            attention_scores = tf.where(mask == 0, -1e9, attention_scores)
            
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return tf.matmul(attention_scores, value), attention_scores
    def call(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = tf.transpose(tf.reshape(query, (tf.shape(query)[0], -1, self.num_heads, self.d_k)), perm=(0, 2, 1, 3)) # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        key = tf.transpose(tf.reshape(key, (tf.shape(key)[0], -1, self.num_heads, self.d_k)), perm=(0, 2, 1, 3)) # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        value = tf.transpose(tf.reshape(value, (tf.shape(value)[0], -1, self.num_heads, self.d_k)), perm=(0, 2, 1, 3)) # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, dropout=self.dropout)
        
        #COmbine heads together again
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (tf.shape(x)[0], -1, self.d_model))
        
        return self.w_o(x)
    
class ResidualConnection(layers.Layer):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = layers.Dropout(dropout)
        self.norm = LayerNormalization(d_model)
        
    def call(self, x, sublayer):
        return x +  self.dropout(sublayer(self.norm(x)))
    
class DecoderBlock(layers.Layer):
    def __init__(self, d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block 
        self.feed_forward_block = feed_forward_block
        self.residual_connections = [ResidualConnection(d_model, dropout) for _ in range(3)]
        
    def call(self, x, encoder_output, decoder_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(layers.Layer):
    def __init__(self, d_model, embed, pos, layers, proj):
        super().__init__()
        self.layers = layers
        self.embed = embed
        self.pos = pos
        self.norm = LayerNormalization(d_model)
        self.proj = proj #intergrate into the decoder block?
        
    def call(self, x, encoder_output, decoder_mask):
        x = self.embed(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, encoder_output, decoder_mask)    
        return self.proj(self.norm(x))
    
class ProjectionLayer(layers.Layer):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = layers.Dense(vocab_size)
        
    def forward(self, x):
        return self.proj(x)
    
def build_decoder(vocab_size, seq_len, d_model=512, num_hiddens=6, num_heads=8, dropout=0.1, d_ff=2048) -> Decoder:
    # vocab_size: number of unique words in captions
    # seq_len: max number or words in 1 sequence
    # d_model: dimension of tokens (number of ints)
    # num_hiddens: number of decoder hidden layers
    # num_heads: number of head in MultiHeadAttention layer 
    # dropout: probability of dropout 
    # d_ff: dimension of FeedForward layer (number of hidden units)
    
    # creating embedding layer
    embed = InputEmbeddings(d_model, vocab_size)
    
    # creating positional layer
    pos = PositionalEncoding(d_model, seq_len, dropout)
    
    # create decoder blocks
    decoder_blocks = []
    for _ in range(num_hiddens):
        self_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        
        decoder_block = DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward_block, dropout)
        
        decoder_blocks.append(decoder_block)
        
    proj = ProjectionLayer(d_model, vocab_size)
    decoder = Decoder(d_model, embed, pos, decoder_blocks, proj)
    
    for p in decoder.trainable_variables:
        if p.shape.rank > 1:
            tf.keras.initializers.glorot_uniform()(p)

    return decoder     