# Gerencia arquivos
import io
import os
from glob import glob
import cv2
import skimage.io as skio
import tensorflow as tf

files_tif_mask = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test/manually_mask/'
files_jpeg_mask = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test/jpeg/masks/'
files_tif_img = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test/patches/'
files_jpeg_img = '/Users/mateus.miranda/INPE-CAP/mestrado/Projeto/officials_projects/ai4luc/data/cerradata/test/jpeg/images/'

# Building Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'building/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/building/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# Building IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'building/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/building/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)


# CA Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'cultivated_area/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/cultivated_area/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# CA IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'cultivated_area/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/cultivated_area/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)


# FF Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'forest/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/forest/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# FF IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'forest/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/forest/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)


# NO Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'non_observed/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/non_observed/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# NO IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'non_observed/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/non_observed/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)

# OU Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'other_uses/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/other_uses/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# OU IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'other_uses/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/other_uses/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)

# PA Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'pasture/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/pasture/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# PA IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'pasture/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/pasture/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)

# SA Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'savanna_formation/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/savanna_formation/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# SA IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'savanna_formation/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/savanna_formation/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)

# WT Mask
for i in glob(os.path.join(os.getcwd(), files_tif_mask+'water/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_mask+'/water/'+nameFile+'.jpeg'
    print(nameFile)
    patch = skio.imread(i)
    cv2.imwrite(path, patch)

# WT IMG
for i in glob(os.path.join(os.getcwd(), files_tif_img+'water/*.tif')):
    nameFile = os.path.basename(i).split('.tif')[0]
    path = files_jpeg_img+'/water/'+nameFile+'.jpeg'
    print(nameFile)
    patch = tf.keras.utils.load_img(i, color_mode='rgb', target_size=(256, 256))
    tf.keras.utils.save_img(path, patch)

