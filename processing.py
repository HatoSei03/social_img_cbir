import random
import numpy as np
import cv2 as cv
import os as osys

from annoy import AnnoyIndex

from tensorflow import random as tfrand
from keras._tf_keras.keras.applications import InceptionResNetV2
from keras._tf_keras.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.applications.inception_resnet_v2 import preprocess_input

def set_random_seed(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    tfrand.set_seed(seed_value)


def define_model():
    # Load the InceptionResNetV2 model without the top classification layer
    base_model = InceptionResNetV2(
        weights='imagenet', include_top=False, input_shape=(500, 500, 3))

    # Add new layers for feature extraction
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # Freeze pre-trained layers
    for layer in base_model.layers:
        try:
            layer.trainable = False
        except:
            pass
    model = Model(inputs=base_model.input, outputs=x)
    return model


def create_annoy_index(features, n_trees=100):
    # Initialize Annoy index
    index = AnnoyIndex(features.shape[1], 'euclidean')

    # Add feature vectors to the index
    for i, feature in enumerate(features):
        index.add_item(i, feature)

    # Build Annoy index
    index.build(n_trees)
    return index


def search_image(query_feature, index, image_names, top_k=10):
    # Search for the top K nearest neighbors
    nearest_indices = index.get_nns_by_vector(query_feature, top_k)
    # Retrieve the corresponding image names
    nearest_images = [image_names[i] for i in nearest_indices]
    return nearest_images


def preprocess_query_image(img_array):
    img_array = cv.resize(img_array, (500, 500))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array
