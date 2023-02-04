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
import math
import statistics
from scipy import stats
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import imgaug.augmenters as iaa

# Morphology and Filtering
import skimage.io as skio
import skimage.color as skc
import skimage.transform as skt
from skimage import img_as_float, color
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, erosion, opening, diameter_closing
from skimage.morphology import square, convex_hull_image, convex_hull_object, disk
from skimage.color import label2rgb

# Machine & Deep Learning
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Py imports
import tools


# -------- CerraNetv3 --------
def cerranet_module(data):
    # Model rgba_data
    path_model = '../AIModels/trained_models/cerranetv3_4rgba_45e_onlytrain_t4_Best.hdf5'
    # Starting the model
    cerranetv3 = load_model(path_model)
    # Data Visualization
    # print('Data shape: ', np.shape(data))

    # Prediction
    pred = cerranetv3.predict(data)
    # Take the maxximun probability of the prediction
    pred_argmax = np.argmax(pred, axis=1)

    # Return the id of the class (range 0 to up 7)
    return pred_argmax
