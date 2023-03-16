"""
Title: Image segmentation with a U-Net-like architecture
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/20
Last modified: 2020/04/20
Description: Image segmentation model trained from scratch on the Oxford Pets dataset.
Accelerator: GPU

Adaptation by: Mateus de Souza Miranda
Last mofied: 2023/02/01
"""

# --------- Packs ---------
# Directory
import os
from glob import glob

# Data
import cv2
import numpy as np
from PIL import Image
import skimage.io as skio
import imgaug.augmenters as iaa

# Graph
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Ml and Dl
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint

# --------- DeepLabv3plus ---------
# Set up
EPOCHS = 48
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_CLASSES = 8
DATA_DIR = '../../../data/cerradata/jpeg_train_subset/'
NUM_TRAIN_IMAGES = 59400    # 75%
NUM_VAL_IMAGES = 19800  # 25%

train_images = sorted(glob(os.path.join(DATA_DIR, "images/**/*.jpeg")))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, "masks/**/*.jpeg")))[:NUM_TRAIN_IMAGES]
val_images = sorted(glob(os.path.join(DATA_DIR, "images/**/*.jpeg")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]
val_masks = sorted(glob(os.path.join(DATA_DIR, "masks/**/*.jpeg")))[
    NUM_TRAIN_IMAGES : NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
]

def read_image(image_path, mask=False):
    image = tf.io.read_file(image_path)
    if mask:
        image = tf.image.decode_jpeg(image, channels=1)
        image.set_shape([None, None, 1])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
    else:
        image = tf.image.decode_jpeg(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
        image = tf.keras.applications.resnet50.preprocess_input(image)
    return image


def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset


train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)


# U-Net model
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=(img_size, img_size, 3))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = keras.layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the unet_model
    unet_model = keras.Model(inputs, outputs)
    return unet_model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(IMAGE_SIZE, NUM_CLASSES)
model.summary()

# Configure the model for training.
# We use the "sparse" version of categorical_crossentropy
# because our target data is integers.
loss = keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=["accuracy"], loss=loss)

path_trained_models = '../../trained_models/'
checkpoint_all = ModelCheckpoint(path_trained_models+"best_unet.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

# Train the model, doing validation at the end of each epoch.
model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=checkpoint_all)
