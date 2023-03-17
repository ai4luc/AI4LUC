"""

"""

# -------- Library --------
# Machine Learning
import torch
import torch.nn as nn
from torch import Tensor

import segmentation_models_pytorch as smp

# Data
import cv2
import rasterio as rio
import imgaug.augmenters as iaa
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tifffile import imread
import numpy as np
import gc
import imghdr
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# Graph
from typing import Any, Callable, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directory manager
import os
import glob
from copy import deepcopy
from pathlib import Path
import copy
import time

# Metrics
from sklearn.metrics import precision_score, f1_score

# -------- Set up --------
# Classes
lista_class = {'building': 0, 'cultivated_area': 1, 'forest': 2, 'non_observed': 3, 'other_uses': 4,
               'pasture': 5, 'savanna_formation': 6, 'water': 7}

lista_class_r = {value: key for key, value in zip(lista_class.keys(), lista_class.values())}
# Constants
IMG_HEIGHT = 256
IMG_WIDTH = 256
NUM_CLASSES = 8
BATCH_SIZE = 100
EPOCHS = 40
PATIENCE = int(EPOCHS * (10 / 100))
exp_directory = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/' \
                'data/cerradata/test/models_prediction/deeplabv3plus_mask/water/'

# GPU config
device = torch.device('mps')
print("Using device", device)


# -------- Metrics --------
def compute_iou(label, pred):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels)
    iou = {}

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I = float((label_i & pred_i).sum())
        U = float((label_i | pred_i).sum())
        iou[lista_class_r[val]] = (I / U)

    return iou


def normalize0to1(imgs):
    norm_img = []
    for img in imgs:
        # Transpose

        # Normalize
        max_img = np.max(img)
        min_img = np.min(img)
        normalized_img = (img - min_img) / (max_img - min_img)

        # Crop
        cropcenter = iaa.CenterCropToFixedSize(height=256, width=256)
        cropped_raster = cropcenter.augment_image(normalized_img)

        norm_img.append(cropped_raster)

    return norm_img


# ------------- Load data -------------
def load_patches(path):
    # Create the sequence of the images path sat[0]
    image_path = path + "/images/water/*.tif"
    # Take full path
    image_names = glob.glob(image_path)
    # File sort
    image_names.sort()
    print(len(image_names), 'Images to test')

    # Create the sequence of the mask path sat[1]
    label_path = path + "/manually_masks/water/*.tif"
    # Take full path
    label_names = glob.glob(label_path)
    # File sort
    label_names.sort()
    print(len(label_names), 'Masks to test')

    # lists
    image_patches = []
    images_normalized = []
    label_patches = []

    # Reading images path
    for name in image_names:
        with rio.open(name) as raster:
            bands = []
            for i in range(1, 4 + 1):
                band = raster.read(i)
                # Crop
                cropcenter = iaa.CenterCropToFixedSize(height=256, width=256)
                cropped_raster = cropcenter.augment_image(band)
                bands.append(cropped_raster)
            image_patches.append(bands)

    # Normalizing 0 to 1
    imagenor_patches = normalize0to1(image_patches)

    # Reading labels path
    for name in label_names:
        with rio.open(name) as raster:
            # Read label
            label = raster.read(1)
            # Crop
            cropcenter = iaa.CenterCropToFixedSize(height=256, width=256)
            cropped_raster = cropcenter.augment_image(label)
            label_patches.append(cropped_raster)

    # To tensor
    imagenor_patches = torch.tensor(imagenor_patches, dtype=torch.float32)
    label_patches = torch.tensor(label_patches, dtype=torch.int32)

    return imagenor_patches, image_names


# Load images and labels together in only on dataset
class SegmentationData(Dataset):
    def __init__(self, image_patches, label_patches):
        self.image_patches = image_patches
        self.label_patches = label_patches

    def __len__(self):
        return len(self.label_patches)

    def __getitem__(self, index):
        return self.image_patches[index], self.label_patches[index]


def unet_model(num_channel: int, num_classes: int):
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                    in_channels=num_channel, classes=num_classes,
                    activation='softmax')

    return unet

# DeepLabv3plus
def deeplabv3plus_model(num_channel: int, num_classes: int):
    deeplabv3plus = smp.DeepLabV3Plus(encoder_name='resnet50', encoder_weights=None,
                                      in_channels=num_channel, classes=num_classes, activation='softmax')

    return deeplabv3plus


def main(path_trained_model, data):
    y_pred = []
    y_true = []

    # Model's architecture
    deeplabv3plus = deeplabv3plus_model(num_channel=4, num_classes=8)

    # Trained model call
    deeplabv3plus.load_state_dict(torch.load(path_trained_model))
    deeplabv3plus.eval()

    deeplabv3plus.to(device)

    with torch.no_grad():
        # Data loading to GPU
        inputs = data.to(device)

        # Prediction
        y_pred = deeplabv3plus(inputs)

    return y_pred


cerradata = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test'
path_model = '../AIModels/trained_models/deeplabv3plus/cerradatav3_run1_DeepLabv3plus.pt'


# Load images and Labels
image_test, image_names = load_patches(path=cerradata)

# Getting images and labels together
data_test = SegmentationData(image_test, image_names)

# Setting up the test dataset
dataset_test = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


for image, image_names in dataset_test:
    # Predicitons
    y_pred = main(path_model, data=image)

    # CPU transfering
    y_pred_argmax = y_pred.argmax(1).cpu().numpy()

    # Saving y_pred
    for pred in range(len(y_pred_argmax)):
        name_file = os.path.basename(image_names[pred])
        cv2.imwrite(exp_directory + 'mask_deeplabv3+_' + str(name_file), y_pred_argmax[pred])


""" 



# CPU transfering
y_pred = y_pred_cRGB.argmax(1).cpu().numpy()
y_true = y_true_cRGB.cpu().numpy()


# Saving the y_pred
#cv2.imwrite(exp_directory + '/maskunet_' + str(i) + '.tif',ypred)
"""

