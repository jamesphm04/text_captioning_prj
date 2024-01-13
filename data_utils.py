from pathlib import Path
from datasets import Dataset

import pandas as pd
import tensorflow as tf

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from dataset import ImageCaptionDataset

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from pickle import dump, load


def extract_features(directories):
    if not os.path.exists('./features.pkl'):    
        data_dir = './data/Images/'
        model = DenseNet201()
        fe = Model(inputs=model.inputs, outputs=model.layers[-2].output)

        img_size = 224
        features = {}
        for image in tqdm(directories): #TODO remove [:10]
            img = load_img(os.path.join(data_dir, image), target_size=(img_size, img_size))
            img = img_to_array(img)
            img = img/255
            img = np.expand_dims(img, axis=0)
            feature = fe.predict(img)
            features[image] = feature
        
        #save features
        dump(features, open('features.pkl', 'wb'))
        
        return features
    else:
        features = load(open('features.pkl', 'rb'))
        features = {k : v for k, v in features.items() if k in directories}
        return features


def caption_preprocessing(df):
    df['caption'] = df['caption'].apply(lambda x: x.lower())
    df['caption'] = df['caption'].apply(lambda x: x.replace('[^A-Za-z]', ''))
    df['caption'] = df['caption'].apply(lambda x: x.replace('\s+', ' '))
    df['caption'] = df['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))
    return df

def get_all_captions(ds):
    for item in ds:
        yield item['caption']

def get_or_build_tokenizer(ds, config):
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

def get_ds(config):
    
    df = pd.read_csv('./data/captions.csv') 
    
    captions = df['caption'].tolist()
    images = df['image'].unique().tolist()
    
    nimages = len(images)
    
    split_index = round(0.9 * nimages)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    df_train = df[df['image'].isin(train_images)]
    df_val = df[df['image'].isin(val_images)]
    
    # get tokenizers
    ds_raw = Dataset.from_pandas(df)
    tokenizer = get_or_build_tokenizer(ds_raw, config)
    # get features
    train_features = extract_features(df_train['image'].values.tolist())
    val_features = extract_features(df_val['image'].values.tolist())
    # get datasets
    train_dataset = ImageCaptionDataset(df_train, 'image', 'caption', config['batch_size'], tokenizer, config['max_length'], train_features)
    val_dataset = ImageCaptionDataset(df_val, 'image', 'caption', config['batch_size'], tokenizer, config['max_length'], val_features)
    
    # Find the maximum length of each sentence in the target sentence
    max_length = 0

    for i, row in df.iterrows():
        caption = row['caption']
        ids = tokenizer.encode(str(caption)).ids
        max_length = max(max_length, len(ids))

    print(f'Max length of target sentence: {max_length}')
    
    # train_dataloader = tf.data.Dataset.from_tensor_slices(train_dataset)
    # val_dataloader = tf.data.Dataset.from_tensor_slices(val_dataset)
    
    return train_dataset, val_dataset, tokenizer
