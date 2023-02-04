"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# ********** VGG16 + SVM **********

# ------ Libs ------
# Directory
import os
from glob import glob

# ML and DL
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input

# Metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Data
from PIL import Image
import numpy as np
from keras.preprocessing import image

# ------ VGG16 + SVM ------

# Lists
x_train = list()
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

# Paths
#data_train = '/scratch/ideeps/mateus.miranda/ai4luc/cerradata80k_splited/train4/'
#data_test = '/scratch/ideeps/mateus.miranda/ai4luc/cerradata80k_splited/test/'
data_train = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test/patches'
data_test = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test/patches'

classes_cerradata = ['building', 'cultivated_area', 'forest', 'non_observed',
                     'other_uses', 'pastur', 'savanna_formation', 'water']

# Data Loading
img = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# X Training
train = img.flow_from_directory(data_train,
                                target_size=(256, 256),
                                batch_size=128,
                                color_mode='rgba',
                                class_mode="categorical")
# Y Training
ytrain = train.classes

# VGG16: Extracting Data Features
# Load the VGG16 model to feature extractor
model_vgg16 = keras.applications.vgg16.VGG16(weights=None, input_shape=(256, 256, 4), pooling='avg', include_top=True)

print('Feature Extraction of training subset...')
for xtrain, ytrain in train:
    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(xtrain)
    # VGG16 does the predictions
    Xfeature = model_vgg16.predict(patch_process)
    # Feature Storage
    x_train.append(Xfeature)

# Getting the list size
n_samples, nx, ny = np.shape(x_train)
# Reshape the list size
train_subset = np.reshape(x_train, (n_samples, nx * ny))

# SVM model: Training
print('SVM training...')
model_svm = svm.SVC().fit(train_subset, ytrain)

print('Training accomplished.\n Assessing model...')


# SVM model: Testing
# BUILDING SUBSET
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
test_subset_path = glob(data_test+'building/*.tif')
test_subset_path.sort()
for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = tf.keras.applications.resnet50.preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(0)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

bl = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# CULTIVATED AREA SUBSET
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
test_subset_path = glob(data_test + 'cultivated_area/*.tif')
test_subset_path.sort()
for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(1)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

ca = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# FOREST SUBSET
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
test_subset_path = glob(data_test + 'forest/*.tif')
test_subset_path.sort()
for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(2)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

ff = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# NON OBSERVED SUBSET
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
test_subset_path = glob(data_test + 'non_observed/*.tif')
test_subset_path.sort()
for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(3)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

no = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# OTHER USES SUBSET
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
test_subset_path = glob(data_test + 'other_uses/*.tif')
test_subset_path.sort()

for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(4)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

ou = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# PASTURE SUBSET
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
test_subset_path = glob(data_test + 'pasture/*.tif')
test_subset_path.sort()

for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(5)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

pa = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# SAVANNA FORMATION SUBSET
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
test_subset_path = glob(data_test + 'savanna_formation/*.tif')
test_subset_path.sort()

for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(6)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

sa = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

# WATER SUBSET
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
test_subset_path = glob(data_test + 'water/*.tif')
test_subset_path.sort()

for raster_path in test_subset_path:
    # Taking name class
    class_name = os.path.basename(os.path.dirname(raster_path))
    # Reading image
    patch = image.load_img(raster_path, target_size=(256, 256), color_mode='rgba')
    array_patch = image.img_to_array(patch)
    normalized_patch = tf.keras.layers.Rescaling(scale=1 / 255)(array_patch)
    expand_patch = np.array(np.expand_dims(normalized_patch, axis=0))
    patch_x = np.copy(expand_patch)

    # Preprocesses a tensor or Numpy array encoding a batch of images
    patch_process = preprocess_input(patch_x)
    # VGG16 does the predictions
    features = model_vgg16.predict(patch_process)

    # Classifying
    y_pred = model_svm.predict(features)
    y_predict.append(y_pred)

    y_true.append(7)
    if y_pred == 0:
        a = a + 1
    elif y_pred == 1:
        b = b + 1
    elif y_pred == 2:
        i = i + 1
    elif y_pred == 3:
        j = j + 1
    elif y_pred == 4:
        k = k + 1
    elif y_pred == 5:
        m = m + 1
    elif y_pred == 6:
        n = n + 1
    elif y_pred == 7:
        o = o + 1

wt = ['BL:', a, 'CA:', b, 'FF:', i, 'NO:', j, 'OU:', k, 'PA:', m, 'SA:', n, 'WT:', o]

print('Test accomplished.')

print('Analysing the results...')
# Assessment
f1score = f1_score(y_true, y_predict, average='weighted')
acc = accuracy_score(y_true, y_predict)
class_report = classification_report(y_true, y_predict, target_names=classes_cerradata)

# Report making
print('Scores:')
print('F1-Score:')
print(f1score)
print('Accuracy:')
print(acc)
print('Accuracy per class:')
print(class_report)

print('Classifications per subsets:')
print('Bl:', bl)
print('Ca:', ca)
print('Ff:', ff)
print('No:', no)
print('Ou:', ou)
print('Pa:', pa)
print('Sa:', sa)
print('Wt:', wt)

path_report = '/scratch/ideeps/mateus.miranda/ai4luc/reports/'
with open(path_report+'SVM_Overall_results.text', 'w', encoding='utf-8') as f:
    f.write('Overall CerraNetv3 performance report \n')
    f.write('1. F1-Score: \n')
    f.write(str(f1score))
    f.write('\n2. Accuracy:\n')
    f.write(str(acc))
    f.write('\n3. Score per class:\n')
    f.write(str(class_report))
    f.write('4. All classification per subset:\n')
    f.write(str(bl))
    f.write(str(ca))
    f.write(str(ff))
    f.write(str(no))
    f.write(str(ou))
    f.write(str(pa))
    f.write(str(sa))
    f.write(str(wt))


