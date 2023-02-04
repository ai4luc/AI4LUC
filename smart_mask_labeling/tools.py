"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# -------- Library --------
# Directory manage
import os
from glob import glob

# Data processing
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import imgaug.augmenters as iaa

# Morphology and Filtering
import skimage.io as skio
from skimage import color

# Machine & Deep Learning
import tensorflow as tf


# -------- Tools --------

# Crop Center Polygon
def clipping_window_over_polygon(data, step_size, window_size):  # V2
    # List
    croped_polygon = []
    # Image to array
    data = np.array(data)

    # Row
    for y in range(0, data.shape[0], step_size):
        # Column
        for x in range(0, data.shape[1], step_size):
            # Sliding windown in the image per pixel
            window = data[y:y + window_size[1], x:x + window_size[0]]

            # Calculate percent over non zeros values
            allpixel = window.shape[0] * window.shape[1]
            nonzeros = np.count_nonzero(window) / 4
            porcent_nonnzeros = (nonzeros / allpixel) * 100

            # If bigger than 92% so take the image's coordinates
            if porcent_nonnzeros > 92 and window.shape[0] > 40 and window.shape[1] > 40:
                # return window # Return only one first for each polygon
                croped_polygon.append(window)

    return croped_polygon


# Generate an empty image and fill it with rgba_patch of the polygon
def fillaimage(data):
    # The width and height of the background tile
    img_w, img_h = data.size
    # Creates a new empty image, RGB mode
    new_img = Image.new('RGBA', (256, 256))
    # The width and height of the new image
    w, h = new_img.size

    # Iterate through a grid
    for row in range(0, w, int(img_w)):
        for colum in range(0, h, int(img_h)):
            # Paste the image at location row, column
            new_img.paste(data, (row, colum))

    return new_img


def save_mask(masks, file_address, address_to_save, classify_mask: False):
    item = 0
    # Get name of the file
    nameFile = os.path.basename(file_address).split('.tif')

    # When true
    if classify_mask:
        cv2.imwrite(address_to_save + '/mask_' + nameFile[0] + '.tif',
                    tf.keras.utils.img_to_array(masks))
    # When False
    else:
        # Load every unlabeled mask
        for mask in masks:
            cv2.imwrite(address_to_save + '/mask_' + str(item) + '_' + nameFile[0] + '.tif',
                        tf.keras.utils.img_to_array(mask))
            item = item +1


# --------  Dataset --------
def rgb_to_gray(rgba_data):
    # RGA to RGB to GRAY
    sample1_gray = color.rgb2gray(color.rgba2rgb(rgba_data))
    # Crop center
    cropcenter = iaa.CenterCropToFixedSize(height=256, width=256)
    cropped_raster = cropcenter.augment_image(np.array(sample1_gray))
    # Storage
    gray_patch = cropped_raster
    # Return a gray pacth
    return gray_patch


def load_image(path):
    # List
    patch = []

    # Path of the Images
    list_file = glob(path)
    # Sorting name file
    list_file.sort()

    # Storage file address
    address = list_file

    # Loop
    for sample in list_file:
        # Read Image
        raster = skio.imread(sample)
        # Crop center
        cropcenter = iaa.CenterCropToFixedSize(height=256, width=256)
        cropped_raster = cropcenter.augment_image(np.array(raster))
        # Array to image
        patch.append(Image.fromarray(cropped_raster))

    return patch, address


def normalize_raster(data):  # V1
    # Image to array
    raster_arr = tf.keras.utils.img_to_array(data)

    img_shape = np.shape(raster_arr)

    if img_shape[1] == 257:
        # Normalize
        normalized_raster = tf.keras.layers.Rescaling(scale=1 / 255)(raster_arr)
        raster_exp_dims = np.expand_dims(normalized_raster, axis=0)

        # Crop center
        tensor_patch = tf.keras.layers.CenterCrop(height=256, width=256)(raster_exp_dims)

    else:
        # Normalize
        normalized_raster = tf.keras.layers.Rescaling(scale=1 / 255)(raster_arr)
        raster_exp_dims = np.expand_dims(normalized_raster, axis=0)
        # To Tensor
        tensor_patch = tf.convert_to_tensor(raster_exp_dims)

    # print('Shape Normalized image: ', np.shape(tensor_patch))
    return tensor_patch
