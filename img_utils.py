from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import numpy as np
import os

def extract_features(directories):
    data_dir = './data/Images/'
    model = DenseNet201()
    fe = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    img_size = 224
    features = {}
    for image in tqdm(directories[:10]):
        img = load_img(os.path.join(data_dir, image), target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img/255
        img = np.expand_dims(img, axis=0)
        feature = fe.predict(img)
        features[image] = feature
        
    return features