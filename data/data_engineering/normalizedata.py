import matplotlib.pyplot as plt
import numpy as np
# Morphology and Filtering
import skimage.io as skio
from skimage import color

# Machine & Deep Learning
import tensorflow as tf
import rasterio as rio


def normalize0to1(img):
  max_img = np.max(img)
  min_img = np.min(img)
  normalized_img = (img - min_img)/(max_img - min_img)
  return normalized_img

path = '../cerradata/jpeg_train_subset/images/building/20200428_209_132_L4_11196.jpeg'

# Read Image
image_patches = []
with rio.open(path) as raster:
  bands = []
  for i in range(1, 4 + 1):
    band = raster.read(i)
    norma_band = normalize0to1(band)
    bands.append(norma_band)
  image_patches.append(bands)

print('Normalized image 1: ', image_patches)
print(np.shape(image_patches))
plt.imshow(image_patches)
plt.show()

raster = skio.imread(path)
raster_arr = np.array(raster).astype('float32')

nor1 = normalize0to1(raster_arr)


normalized_raster = tf.keras.layers.Rescaling(scale=1 / 255)(raster_arr)
raster_exp_dims = np.expand_dims(normalized_raster, axis=0)
print('Normalized image 2: ', raster_exp_dims)
