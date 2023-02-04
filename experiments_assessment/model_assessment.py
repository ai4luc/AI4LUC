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
import os
import numpy as np
import pandas as pd
from glob import glob
from tensorflow import keras
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Arrays
y_predict = []
y_true = []
overall_acc_model = []
overall_f1_model = []
overall_classes = []
overall_perclass = []
bl = []
ca = []
ff = []
no = []
ou = []
pa = []
sa = []
wt = []

classes_cerradata = ['building', 'cultivated_area', 'forest', 'non_observed',
                     'other_uses', 'pastur', 'savanna_formation', 'water']

# Loading trained_models and weight
print('Connecting ...')
path_model = glob('../AIModels/trained_models/*.hdf5')
# Reading models
for model in path_model:
    # Starting model
    cerranet = load_model(model)

    # Taking name of the models
    name_model = os.path.basename(model).split('.hdf5')
    print(name_model[0] + ' model connected.')

    # Loading dataset

    # ----- BUILDING SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/building/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(0)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    bl = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m,'SA:', n, 'WT:', o]

    # ----- CULTIVATED AREA SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/cultivated_area/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(1)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    ca = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m,'SA:', n, 'WT:', o]

    # ----- FOREST SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/forest/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(2)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    ff = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m,'SA:', n, 'WT:', o]

    # ----- NON OBSERVED SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/non_observed/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(3)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    no = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

    # ----- OTHER USES SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/other_uses/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(4)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    ou = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

    # ----- PASTURE SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/pasture/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(5)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    pa = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

    # ----- SAVANNA SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/savanna_formation/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(6)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    sa = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m,'SA:', n, 'WT:', o]

    # ----- WATER SET -----
    # Counters
    a = 0
    b = 0
    i = 0
    j = 0
    k = 0
    m = 0
    n = 0
    o = 0

    # Path Dataset
    path_dataset = glob('../data/cerradata/test/patches/water/*.tif')
    path_dataset.sort()
    for raster_path in path_dataset:
        # Taking name class
        class_name = os.path.basename(os.path.dirname(raster_path))
        # Reading image
        patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
        x_test = image.img_to_array(patch)
        x_test = tf.keras.layers.Rescaling(scale=1 / 255)(x_test)
        x_test = np.expand_dims(x_test, axis=0)
        # Classifying
        pred_y = cerranet.predict(x_test)
        y_pred_argmax = np.argmax(pred_y, axis=1)
        y_predict.append(y_pred_argmax)

        y_true.append(7)
        if y_pred_argmax == 0:
            a = a + 1
        elif y_pred_argmax == 1:
            b = b + 1
        elif y_pred_argmax == 2:
            i = i + 1
        elif y_pred_argmax == 3:
            j = j + 1
        elif y_pred_argmax == 4:
            k = k + 1
        elif y_pred_argmax == 5:
            m = m + 1
        elif y_pred_argmax == 6:
            n = n + 1
        elif y_pred_argmax == 7:
            o = o + 1
    wt = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m,'SA:', n, 'WT:', o]

    # Assessment
    f1score = f1_score(y_true, y_predict, average='weighted')
    overall_f1_model.append(f1score)
    acc = accuracy_score(y_true, y_predict)
    overall_acc_model.append(acc)
    class_report = classification_report(y_true, y_predict, target_names=classes_cerradata)
    overall_classes.append(class_report)

# -------  Statistics -------
# Standard deviation
overall_std_f1 = np.std(overall_f1_model)
overall_std_acc = np.std(overall_acc_model)
# Mean
overall_mean_f1 = np.mean(overall_f1_model)
overall_mean_acc = np.mean(overall_acc_model)

# Report making
print('Scores:')
print('F1-Score:')
print(overall_f1_model)
print('Accuracy:')
print(overall_acc_model)
print('Accuracy per class:')
print(overall_classes[0])

print('Overall Statistics')
print('STD:')
print(overall_std_f1)
print(overall_std_acc)
print('Mean:')
print(overall_mean_f1)
print(overall_mean_acc)

print('Classifications per subsets:')
print('Bl:', bl)
print('Ca:', ca)
print('Ff:', ff)
print('No:', no)
print('Ou:', ou)
print('Pa:', pa)
print('Sa:', sa)
print('Wt:', wt)

with open('CerraNetv3 Overall results.csv', 'w', encoding='utf-8') as f:
    f.write('Overall CerraNetv3 performance report \n')
    f.write('1. Stand deviation F1-Score: \n')
    f.write(str(overall_std_f1))
    f.write('\n2. Mean F1-Score:\n')
    f.write(str(overall_mean_f1))
    f.write('\n3. Score per class:\n')
    f.write(str(overall_classes))
    f.write('4. All classification per subset:\n')
    f.write(str(bl))
    f.write(str(ca))
    f.write(str(ff))
    f.write(str(no))
    f.write(str(ou))
    f.write(str(pa))
    f.write(str(sa))
    f.write(str(wt))
