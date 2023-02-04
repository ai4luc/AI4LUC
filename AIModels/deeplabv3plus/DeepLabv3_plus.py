"""
Title: Multiclass semantic segmentation using DeepLabV3+
Author: [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2021/08/31
Last modified: 2021/09/1
Description: Implement DeepLabV3+ architecture for Multi-class Semantic Segmentation.

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

# Graph
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Ml and Dl
import tensorflow as tf
from tensorflow import keras

# --------- DeepLabv3plus ---------
# Set up
IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_CLASSES = 8
DATA_DIR = '../../data/cerradata/test/'
NUM_TRAIN_IMAGES = 80
NUM_VAL_IMAGES = 20

# List of samples to train
train_images = sorted(glob(os.path.join(DATA_DIR, 'patches_for_dlv3plus/*.tif')))[:NUM_TRAIN_IMAGES]
train_masks = sorted(glob(os.path.join(DATA_DIR, 'masks_for_dlv3plus/*.tif')))[:NUM_TRAIN_IMAGES]
# List of samples to validate
val_images = sorted(glob(os.path.join(DATA_DIR, 'patches_for_dlv3plus/*.tif')))[
             NUM_TRAIN_IMAGES: NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
             ]
val_masks = sorted(glob(os.path.join(DATA_DIR, 'masks_for_dlv3plus/*.tif')))[
            NUM_TRAIN_IMAGES: NUM_VAL_IMAGES + NUM_TRAIN_IMAGES
            ]


def read_image(image_path, mask=False):
    # When been a mask
    image = []
    if mask:
        for patch in image_path:
            masks = skio.imread(patch)
            masks = np.array(masks).astype('int32')
            image.append(image)

    # When been a img
    else:
        for patch in image_path:
            # Read
            image = tf.keras.utils.load_img(patch, color_mode='rgb',
                                            target_size=(256, 256),
                                            interpolation='nearest')
            # Normalize
            #image = tf.keras.utils.img_to_array(image)
            #image = tf.keras.layers.Rescaling(scale=1 / 255)(image)
            #image = np.expand_dims(image, axis=0)
            #image = np.copy(image)
            # Preprocess
            #image = tf.keras.utils.img_to_array(image)
            #image = tf.image.resize(images=image, size=[IMAGE_SIZE, IMAGE_SIZE])
            #image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

test = read_image(train_masks, mask=True)
print(np.unique(test))

"""

def load_data(image_list, mask_list):
    image = read_image(image_list)
    mask = read_image(mask_list, mask=True)
    return image, mask


def data_generator(image_list, mask_list):
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset



# The model
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = keras.layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = keras.layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = keras.layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = keras.layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = keras.layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)


# --------- Application ---------
# Data loading
train_dataset = data_generator(train_images, train_masks)
val_dataset = data_generator(val_images, val_masks)

print("Train Dataset:", train_dataset)
print("Val Dataset:", val_dataset)

# Model loading
model = DeeplabV3Plus(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()

# Training
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=loss,
    metrics=["accuracy"],
)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=40)
"""