# CerraNet v2: assessment of model

# Library
import os
import numpy as np
import pandas as pd
from glob import glob
from tensorflow import keras
from sklearn import metrics
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import model_from_json


# Loading model and weight
print('Connecting vgg16...')
net = open('../modelos_prontos/cerranetv3/cerranetv3_json.json', 'r')
cnnNet = net.read()
net.close()
cerranet = keras.models.model_from_json(cnnNet)
cerranet.load_weights('../modelos_prontos/cerranetv3/cerranetv3_weights.h5')
print('VGG16 online.')

print('Starting performance testing...')

# CHECKING FARMING CLASS
fr = []   # farming
i = 0     # counter
j = 0     # counter
k = 0     # counter
m = 0     # counter
n = 0     # counter

path = '../../data/dataset_cerradov3_NIR+G+B_splited_50k/test/farming/*.tif'
for img in glob(os.path.join(os.getcwd(), path)):
    # Read the images
    ia = image.load_img(img, target_size=(256, 256))
    img_plot = ia

    # Transform them in array
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    # Get image information
    path = os.path.dirname(img)
    file_name = os.path.basename(img)
    origin_address = path + '/' + file_name

    # Count image
    print('Reading Image:', file_name, '...')

    # Make predictions
    forecast = cerranet.predict(ia)
    p = pd.DataFrame(forecast, dtype='float32')

    # Image Discriminator:
    if p.loc[0,0] > p.loc[0,1] and p.loc[0,0] > p.loc[0,2] and p.loc[0,0] > p.loc[0,3] and p.loc[0,0] > p.loc[0,4]:
        i = i+1
        print('farming')

    if p.loc[0,1] > p.loc[0,0] and p.loc[0,1] > p.loc[0,2] and p.loc[0,1] > p.loc[0,3] and p.loc[0,1] > p.loc[0,4]:
        j = j+1
        print('forest_formation')

    if p.loc[0,2] > p.loc[0,0] and p.loc[0,2] > p.loc[0,1] and p.loc[0,2] > p.loc[0,3] and p.loc[0,2] > p.loc[0,4]:
        k = k+1
        print('non_forest_area')

    if p.loc[0,3] > p.loc[0,0] and p.loc[0,3] > p.loc[0,1] and p.loc[0,3] > p.loc[0,2] and p.loc[0,3] > p.loc[0,4]:
        m = m+1
        print('savanna_formation')

    if p.loc[0,4] > p.loc[0,0] and p.loc[0,4] > p.loc[0,1] and p.loc[0,4] > p.loc[0,2] and p.loc[0,4] > p.loc[0,3]:
        n = n+1
        print('water')

fr.append(i)
fr.append(j)
fr.append(k)
fr.append(m)
fr.append(n)

# CHECKING FOREST_FORMATION CLASS
ff = []   # forest_formation
i = 0     # counter
j = 0     # counter
k = 0     # counter
m = 0     # counter
n = 0     # counter

path = '../../data/dataset_cerradov3_NIR+G+B_splited_50k/test/forest_formation/*.tif'
for img in glob(os.path.join(os.getcwd(), path)):
    # Read the images
    ia = image.load_img(img, target_size=(256, 256))
    img_plot = ia

    # Transform them in array
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    # Get image information
    path = os.path.dirname(img)
    file_name = os.path.basename(img)
    origin_address = path + '/' + file_name

    # Count image
    print('Reading Image:', file_name, '...')

    # Make predictions
    forecast = cerranet.predict(ia)
    p = pd.DataFrame(forecast, dtype='float32')

    # Image Discriminator:
    if p.loc[0,0] > p.loc[0,1] and p.loc[0,0] > p.loc[0,2] and p.loc[0,0] > p.loc[0,3] and p.loc[0,0] > p.loc[0,4]:
        i = i+1
        print('farming')

    if p.loc[0,1] > p.loc[0,0] and p.loc[0,1] > p.loc[0,2] and p.loc[0,1] > p.loc[0,3] and p.loc[0,1] > p.loc[0,4]:
        j = j+1
        print('forest_formation')

    if p.loc[0,2] > p.loc[0,0] and p.loc[0,2] > p.loc[0,1] and p.loc[0,2] > p.loc[0,3] and p.loc[0,2] > p.loc[0,4]:
        k = k+1
        print('non_forest_area')

    if p.loc[0,3] > p.loc[0,0] and p.loc[0,3] > p.loc[0,1] and p.loc[0,3] > p.loc[0,2] and p.loc[0,3] > p.loc[0,4]:
        m = m+1
        print('savanna_formation')

    if p.loc[0,4] > p.loc[0,0] and p.loc[0,4] > p.loc[0,1] and p.loc[0,4] > p.loc[0,2] and p.loc[0,4] > p.loc[0,3]:
        n = n+1
        print('water')

ff.append(i)
ff.append(j)
ff.append(k)
ff.append(m)
ff.append(n)

# CHECKING NON_FOREST_AREA CLASS
nfa = []  # non_forest_area
i = 0     # counter
j = 0     # counter
k = 0     # counter
m = 0     # counter
n = 0     # counter

path = '../../data/dataset_cerradov3_NIR+G+B_splited_50k/test/non_forest_area/*.tif'
for img in glob(os.path.join(os.getcwd(), path)):
    # Read the images
    ia = image.load_img(img, target_size=(256, 256))
    img_plot = ia

    # Transform them in array
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    # Get image information
    path = os.path.dirname(img)
    file_name = os.path.basename(img)
    origin_address = path + '/' + file_name

    # Count image
    print('Reading Image:', file_name, '...')

    # Make predictions
    forecast = cerranet.predict(ia)
    p = pd.DataFrame(forecast, dtype='float32')

    # Image Discriminator:
    if p.loc[0,0] > p.loc[0,1] and p.loc[0,0] > p.loc[0,2] and p.loc[0,0] > p.loc[0,3] and p.loc[0,0] > p.loc[0,4]:
        i = i+1
        print('farming')

    if p.loc[0,1] > p.loc[0,0] and p.loc[0,1] > p.loc[0,2] and p.loc[0,1] > p.loc[0,3] and p.loc[0,1] > p.loc[0,4]:
        j = j+1
        print('forest_formation')

    if p.loc[0,2] > p.loc[0,0] and p.loc[0,2] > p.loc[0,1] and p.loc[0,2] > p.loc[0,3] and p.loc[0,2] > p.loc[0,4]:
        k = k+1
        print('non_forest_area')

    if p.loc[0,3] > p.loc[0,0] and p.loc[0,3] > p.loc[0,1] and p.loc[0,3] > p.loc[0,2] and p.loc[0,3] > p.loc[0,4]:
        m = m+1
        print('savanna_formation')

    if p.loc[0,4] > p.loc[0,0] and p.loc[0,4] > p.loc[0,1] and p.loc[0,4] > p.loc[0,2] and p.loc[0,4] > p.loc[0,3]:
        n = n+1
        print('water')

nfa.append(i)
nfa.append(j)
nfa.append(k)
nfa.append(m)
nfa.append(n)


# CHECKING SAVANNA_FORMATION CLASS
sf = []   # savanna_formation
i = 0     # counter
j = 0     # counter
k = 0     # counter
m = 0     # counter
n = 0     # counter

path = '../../data/dataset_cerradov3_NIR+G+B_splited_50k/test/savanna_formation/*.tif'
for img in glob(os.path.join(os.getcwd(), path)):
    # Read the images
    ia = image.load_img(img, target_size=(256, 256))
    img_plot = ia

    # Transform them in array
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    # Get image information
    path = os.path.dirname(img)
    file_name = os.path.basename(img)
    origin_address = path + '/' + file_name

    # Count image
    print('Reading Image:', file_name, '...')

    # Make predictions
    forecast = cerranet.predict(ia)
    p = pd.DataFrame(forecast, dtype='float32')

    # Image Discriminator:
    if p.loc[0,0] > p.loc[0,1] and p.loc[0,0] > p.loc[0,2] and p.loc[0,0] > p.loc[0,3] and p.loc[0,0] > p.loc[0,4]:
        i = i+1
        print('farming')

    if p.loc[0,1] > p.loc[0,0] and p.loc[0,1] > p.loc[0,2] and p.loc[0,1] > p.loc[0,3] and p.loc[0,1] > p.loc[0,4]:
        j = j+1
        print('forest_formation')

    if p.loc[0,2] > p.loc[0,0] and p.loc[0,2] > p.loc[0,1] and p.loc[0,2] > p.loc[0,3] and p.loc[0,2] > p.loc[0,4]:
        k = k+1
        print('non_forest_area')

    if p.loc[0,3] > p.loc[0,0] and p.loc[0,3] > p.loc[0,1] and p.loc[0,3] > p.loc[0,2] and p.loc[0,3] > p.loc[0,4]:
        m = m+1
        print('savanna_formation')

    if p.loc[0,4] > p.loc[0,0] and p.loc[0,4] > p.loc[0,1] and p.loc[0,4] > p.loc[0,2] and p.loc[0,4] > p.loc[0,3]:
        n = n+1
        print('water')

sf.append(i)
sf.append(j)
sf.append(k)
sf.append(m)
sf.append(n)


# CHECKING WATER CLASS
wt = []   # water
i = 0     # counter
j = 0     # counter
k = 0     # counter
m = 0     # counter
n = 0     # counter

path = '../../data/dataset_cerradov3_NIR+G+B_splited_50k/test/water/*.tif'
for img in glob(os.path.join(os.getcwd(), path)):
    # Read the images
    ia = image.load_img(img, target_size=(256, 256))
    img_plot = ia

    # Transform them in array
    ia = image.img_to_array(ia)
    ia /= 255
    ia = np.expand_dims(ia, axis=0)

    # Get image information
    path = os.path.dirname(img)
    file_name = os.path.basename(img)
    origin_address = path + '/' + file_name

    # Count image
    print('Reading Image:', file_name, '...')

    # Make predictions
    forecast = cerranet.predict(ia)
    p = pd.DataFrame(forecast, dtype='float32')

    # Image Discriminator:
    if p.loc[0,0] > p.loc[0,1] and p.loc[0,0] > p.loc[0,2] and p.loc[0,0] > p.loc[0,3] and p.loc[0,0] > p.loc[0,4]:
        i = i+1
        print('farming')

    if p.loc[0,1] > p.loc[0,0] and p.loc[0,1] > p.loc[0,2] and p.loc[0,1] > p.loc[0,3] and p.loc[0,1] > p.loc[0,4]:
        j = j+1
        print('forest_formation')

    if p.loc[0,2] > p.loc[0,0] and p.loc[0,2] > p.loc[0,1] and p.loc[0,2] > p.loc[0,3] and p.loc[0,2] > p.loc[0,4]:
        k = k+1
        print('non_forest_area')

    if p.loc[0,3] > p.loc[0,0] and p.loc[0,3] > p.loc[0,1] and p.loc[0,3] > p.loc[0,2] and p.loc[0,3] > p.loc[0,4]:
        m = m+1
        print('savanna_formation')

    if p.loc[0,4] > p.loc[0,0] and p.loc[0,4] > p.loc[0,1] and p.loc[0,4] > p.loc[0,2] and p.loc[0,4] > p.loc[0,3]:
        n = n+1
        print('water')

wt.append(i)
wt.append(j)
wt.append(k)
wt.append(m)
wt.append(n)



# ASSESSMENT OF MODEL
print('CerraNetv3 Accuracy Assessmet Report:')

# Overall accuracy
sumAllClass = sum(fr) + sum(ff) + sum(nfa) + sum(sf) + sum(wt)
sumCorrectClassification = fr[0] + ff[1] + nfa[2] + sf[3] + wt[4]
overall_accuracy = round((sumCorrectClassification*100)/sumAllClass, 2)

# F1-Score
# tp = true positive
tp = sumCorrectClassification

# fp = false positive
fp_farming = ff[0] + nfa[0] + sf[0] + wt[0]
fp_forest_formation = fr[1] + nfa[1] + sf[1] + wt[1]
fp_non_forest_area = fr[2] + ff[2] + sf[2] + wt[2]
fp_savanna_formation = fr[3] + ff[3] + nfa[3] + wt[3]
fp_water = fr[4] + ff[4] + nfa[4] + sf[4]

fp = fp_farming + fp_forest_formation + fp_non_forest_area + fp_savanna_formation + fp_water

# fn = false negative
fn_farming = fr[1] + fr[2] + fr[3] + fr[4]
fn_forest_formation = ff[0] + ff[2] + ff[3] + ff[4]
fn_non_forest_area = nfa[0] + nfa[1] + nfa[3] + nfa[4]
fn_savanna_formation = sf[0] + sf[1] + sf[2] + sf[4]
fn_water = wt[0] + wt[1] + wt[2] + wt[3]
fn = fn_farming + fn_forest_formation + fn_non_forest_area + fn_savanna_formation + fn_water

# precision metric
precision = tp/(tp+fp)

# recall metric
recall = tp/(tp+fn)

# f1-score
f1Score = (2 * (precision * recall) / (precision + recall))

# Accuracy Farming
accuracyFarming = round(fr[0]*100/sum(fr),2)
print('1 Farming subset')
print('1.1 Accuracy: ', accuracyFarming)
print('1.2 Classifications: ')
print('- Farming: ', fr[0])
print('- Forest_formation: ', fr[1])
print('- Non_forest_area: ', fr[2])
print('- Savanna_formation: ', fr[3])
print('- Water: ', fr[4])
print('- Correct classification: ', fr[0])
print('- Incorrect classification: ', fn_farming)


# Accuracy Forest
accuracyForest = round(ff[1]*100/sum(ff),2)
print('2 Forest_formation subset')
print('2.1 Accuracy: ', accuracyForest)
print('2.2 Classifications: ')
print('- Farming: ', ff[0])
print('- Forest_formation: ', ff[1])
print('- Non_forest_area: ', ff[2])
print('- Savanna_formation: ', ff[3])
print('- Water: ', ff[4])
print('- Correct classification: ', ff[1])
print('- Incorrect classification: ', fn_forest_formation)


# Accuracy Non_Forest_Area
accuracyNonForestArea = round(nfa[2]*100/sum(nfa),2)
print('3 Non_Forest_Area subset')
print('3.1 Accuracy: ', accuracyNonForestArea)
print('3.2 Classifications: ')
print('- Farming: ', nfa[0])
print('- Forest_formation: ', nfa[1])
print('- Non_forest_area: ', nfa[2])
print('- Savanna_formation: ', nfa[3])
print('- Water: ', nfa[4])
print('- Correct classification: ', nfa[2])
print('- Incorrect classification: ', fn_non_forest_area)

# Accuracy Savanna_Formation
accuracySavannaFormation = round(sf[3]*100/sum(sf),2)
print('4 Savanna_Formation subset')
print('4.1 Accuracy: ', accuracySavannaFormation)
print('4.2 Classifications: ')
print('- Farming: ', sf[0])
print('- Forest_formation: ', sf[1])
print('- Non_forest_area: ', sf[2])
print('- Savanna_formation: ', sf[3])
print('- Water: ', sf[4])
print('- Correct classification: ', sf[3])
print('- Incorrect classification: ', fn_savanna_formation)

# Accuracy Water
accuracyWater = round(wt[4]*100/sum(wt),2)
print('5 Water subset')
print('5.1 Accuracy: ', accuracyWater)
print('5.2 Classifications: ')
print('- Farming: ', wt[0])
print('- Forest_formation: ', wt[1])
print('- Non_forest_area: ', wt[2])
print('- Savanna_formation: ', wt[3])
print('- Water: ', wt[4])
print('- Correct classification: ', wt[4])
print('- Incorrect classification: ', fn_water)

# Over all
print('6 Overall performance')
print('6.1 Accuracy: ', overall_accuracy)
print('6.2 Precision: ', precision)
print('6.3 Recall: ', recall)
print('6.4 F1-Score: ', f1Score)



