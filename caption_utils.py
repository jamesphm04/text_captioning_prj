from pathlib import Path
from config import get_config
from datasets import Dataset

import pandas as pd
import tensorflow as tf

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

config = get_config()


def caption_preprocessing(df):
    df['caption'] = df['caption'].apply(lambda x: x.lower())
    df['caption'] = df['caption'].apply(lambda x: x.replace('[^A-Za-z]', ''))
    df['caption'] = df['caption'].apply(lambda x: x.replace('\s+', ' '))
    df['caption'] = df['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    df['caption'] ='[SOS] ' + df['caption'] + ' [EOS]'
    return df

def get_all_captions(ds):
    for item in ds:
        yield item['caption']

def get_or_build_tokenizer(ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_captions(ds), trainer=trainer )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_caption():
    
    df = pd.read_csv('./data/captions.csv') 
    
    captions = df['caption'].tolist()
    images = df['image'].unique().tolist()
    
    nimages = len(images)
    
    split_index = round(0.85 * nimages)
    train_images = images[:split_index]
    test_images = images[split_index:]
    
    train = df[df['image'].isin(train_images)]
    test = df[df['image'].isin(test_images)]
    
    # # Build tokenizers
    ds_raw = Dataset.from_pandas(df)
    tokenizer = get_or_build_tokenizer(ds_raw)
    
    print(tokenizer.encode(captions[1]).ids)
    
    
    
    # num_captions = len(df['caption'])
    
    # print('Len of dataset: ', num_captions)
    
    

    
    # # Keep 90% for training, 10% for validation
    # train_ds_raw = ds_raw[:train_ds_size*5]
    # val_ds_raw = ds_raw[train_ds_size*5:]
    
    
    # return train_ds_raw, val_ds_raw, tokenizer
    # train_ds = NewsSumDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['text_src'], config['text_tgt'], config['src_seq_len'], config['tgt_seq_len'])
    # val_ds = NewsSumDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['text_src'], config['text_tgt'], config['src_seq_len'], config['tgt_seq_len'])
    
    # # Find the maximum length of each sentence in the source and target sentence
    # max_len_src = 0
    # max_len_tgt = 0

    # for item in ds_raw:
    #     src_ids = tokenizer_src.encode(item[config['text_src']]).ids
    #     tgt_ids = tokenizer_tgt.encode(item[config['text_tgt']]).ids
    #     max_len_src = max(max_len_src, len(src_ids))
    #     max_len_tgt = max(max_len_tgt, len(tgt_ids))

    # print(f'Max length of source sentence: {max_len_src}')
    # print(f'Max length of target sentence: {max_len_tgt}')
    
    # train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    # val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    # return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

