"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# -------- Packs --------
# Library
# Dir
import os
from glob import glob

# Data
import numpy as np
import skimage.io as skio

# DL
import tensorflow as tf
from keras.models import load_model

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from keras import metrics


# -------- Model --------
# Loading trained_models and weight
print('Connecting ...')
path_model = glob('../trained_models/unet_8class/best_unet.hdf5')
unet_model = load_model(path_model[0])

# Taking name of the models
name_model = os.path.basename(path_model[0]).split('.hdf5')
print(name_model[0] + ' model connected.')

# -------- Test subset --------
# Set up
BATCH_SIZE = 1
IMAGE_SIZE = 256
NUM_CLASSES = 8

# List
y_pred = []

# Paths
path_testsubset = '../../data/cerradata/test/jpeg/'
test_images = sorted(glob(os.path.join(path_testsubset, "images/**/*.jpeg")))
test_masks = sorted(glob(os.path.join(path_testsubset, "masks/**/*.jpeg")))


def load_mask(path):
    # List
    patch = list()
    # Path of the Images
    list_file = glob(path)
    # Sorting name file
    list_file.sort()

    # Loop
    for sample in list_file:
        # Read Image
        raster = skio.imread(sample)
        patch.append(raster)

    return patch,

# Paths
path_mask = '../../data/cerradata/test/manually_mask/**/*.tif'
y_true = load_mask(path_mask)

# Data loading
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

# Test subset
test_subset = data_generator(test_images, test_masks)

# -------- Predicting and Assessment --------
for img, mask in test_subset:
    # # Predicting
    pred = unet_model.predict(img)
    pred = np.argmax(pred, axis=-1)
    pred = np.expand_dims(pred, axis=-1)
    pred = np.reshape(pred, (256, 256, 1))
    # Saving
    y_pred.append(pred)

y_pred = np.array(y_pred).astype('int32')
y_true = np.array(y_true).astype('int32')

print(np.unique(y_true))
print(np.unique(y_pred))


# Assessing
# Metrics

# 1. F1-score
f1score = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
print('f1-score ', f1score)

# 2. Mean IoU
meanIou = metrics.MeanIoU(num_classes=8)
meanIou.update_state(y_true, y_pred)
result_meanIou = meanIou.result().numpy()
print('Mean iou: ', result_meanIou)

# 3. IoU
iou = metrics.IoU(num_classes=8, target_class_ids=[0,1,2,3,4,5,6,7])
iou.update_state(y_true, y_pred)
result_Iou = iou.result().numpy()
print('iou: ', result_Iou)

# 4. Precision
f1score = precision_score(y_true.flatten(), y_pred.flatten(), average='weighted')
print('precision: ', f1score)

# 5. Recall
f1score = recall_score(y_true.flatten(), y_pred.flatten(), average='weighted')
print('recall ', f1score)

# 6. classification_report
classes_cerradata = ['building', 'cultivated_area', 'forest', 'non_observed',
                     'other_uses', 'pastur', 'savanna_formation', 'water']
class_report = classification_report(y_true.flatten(), y_pred.flatten(), target_names=classes_cerradata)
print(class_report)


