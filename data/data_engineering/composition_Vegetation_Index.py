import numpy as np
import tifffile as tif
from osgeo import gdal
from glob import glob
import os
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import show

# Reading the raw data
data_path = glob(os.path.join(os.getcwd(), '../data/cerradov3_multiBand/tocantins/CBERS_4A_WPM_20211002_210_126_L4/*.tif'))
data_path.sort()

bands = [data_path[0], data_path[1], data_path[2], data_path[3], data_path[4]]

band_pan = gdal.Open(bands[0])
band_pan = band_pan.ReadAsArray()

band_blue = gdal.Open(bands[1])
band_blue = band_blue.ReadAsArray()

band_green = gdal.Open(bands[2])
band_green = band_green.ReadAsArray()

band_red = gdal.Open(bands[3])
band_red = band_red.ReadAsArray()

band_nir = gdal.Open(bands[4])
band_nir = band_nir.ReadAsArray()

# Take Geo information from data
red = gdal.Open(bands[1])
geoMeta = red.GetGeoTransform()
print(geoMeta)

geo_red = rio.open(bands[0])
crs = geo_red.crs
print(crs)

# Calculating the Vegetation Index
# 1 SAVI: use for analyzes of young crops, for arid regions with sparse vegetation
# Constant
L = 1  # If image has hight concentration of vegetation L = 0, else L = 1

# Calculate
savi = ((band_nir - band_red) / (band_nir + band_red + L)) * (1 + L)

# 2 EVI: to analyze areas with tropical forests and preferably with minimal topographic effects.
# Constants
q1 = 1
q2 = 2.5
c1 = 6
c2 = 7.5

# Calculate
evi = q2 * ((band_nir - band_red) / (band_nir + (c1*band_red) - (c2 * band_blue) + q1))

# 3 NDWI: detection of flooded agricultural land; detection of irrigated cultivated land
ndwi = ((band_green - band_nir) / (band_green + band_nir))

# 4 GNDVI: assessment of weakened and aging vegetation and assessment of nitrogen content in plant leaves by multispectral data, which lacks the extreme red channel.
gndvi = ((band_nir - band_green)/(band_nir+band_green))

# 5 NDVI
ndvi = (band_nir - band_red) / (band_nir + band_red)

