"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# -------- Packs --------
# Data
import os
import numpy as np
from glob import glob
import skimage.io as skio

# Metrics
from sklearn.metrics import f1_score, precision_score
from keras import metrics

# Graphs
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# -------- Pixel-based assessment --------
# Set up
classes_cerradata = ['building', 'cultivated_area', 'forest', 'non_observed',
                     'other_uses', 'pasture', 'savanna_formation', 'water']


# Data loading

def load_image(path):
    # List
    patch = list()
    name_file = []
    # Path of the Images
    list_file = glob(path)
    # Sorting name file
    list_file.sort()

    # Loop
    for sample in list_file:
        # name of file
        name_file.append(os.path.basename(sample))

        # Read Image
        raster = skio.imread(sample)
        patch.append(raster)

    return patch, name_file


# Paths
path = '../data/cerradata/test/'
y_true, name_ytrue = load_image(path + 'manually_masks/BNOW/**/*.tif')
y_pred, name_ypred = load_image(path + 'models_prediction/deeplabv3plus_mask/BNOW/**/*.tif')

y_pred = np.array(y_pred).astype('int32')
y_true = np.array(y_true).astype('int32')
print(np.unique(y_true))

# Metrics
"""
# 1. Mean IoU
meanIou = metrics.MeanIoU(num_classes=8)
meanIou.update_state(y_true, y_pred)
result_meanIou = meanIou.result().numpy()
print('Mean IoU:', result_meanIou)

# 2. IoU
iou = metrics.IoU(num_classes=8, target_class_ids=[0, 1, 2, 3, 4, 5, 6, 7])
iou.update_state(y_true, y_pred)
result_Iou = iou.result().numpy()
print('IoU:', result_Iou)
"""

# 3. F1-score
f1score = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted')
print('F1-score:', float(f1score))

# Precision
precision = precision_score(y_true.flatten(), y_pred.flatten(), average='weighted')
print('Precision:', float(precision))

"""
# Check whether there is mask with id > 7
for i in range(len(y_true)):
    verify = np.unique(y_true[i])
    for j in verify:
        if j > 7:
            print(verify)
            print(name_ytrue[i])
"""
