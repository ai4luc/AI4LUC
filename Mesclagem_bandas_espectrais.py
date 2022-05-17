"""
Beta
©MateusMiranda
"""
# Library
import os
from osgeo import gdal
from osgeo import osr
from glob import glob
import matplotlib.pyplot as plt
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio


# Get sample data from the directory
data_path = glob(os.path.join(os.getcwd(), '../data/AMAZONIA_1_WFI_20211122_035_020_L4/*.tif'))
data_path.sort()

print(data_path)

# Select the bands NIR, 
bands = [data_path[2], data_path[1], data_path[0]]
red = gdal.Open(bands[0])
green = gdal.Open(bands[1])
blue = gdal.Open(bands[2])


# Get features of the bands
ny = red.RasterYSize
nx = red.RasterXSize
geo = red.GetGeoTransform()
print(geo)

arr_red = rio.open(bands[0])
crs = arr_red.crs

# Create image stack and apply nodata value for Landsat
name_dir = os.path.basename(bands[0])
output_dir = os.path.join("images_composed", name_dir)
if os.path.isdir(output_dir) == False:
    os.mkdir(output_dir)

name_img = os.path.basename(bands[0])
raster_out_path = os.path.join(output_dir, name_img)

arr_st, meta = es.stack(bands, out_path=raster_out_path)

# Create figure with one plot
fig, ax = plt.subplots(figsize=(5, 5))

# Plot NIR, red, and green bands, respectively, with stretch
#ep.plot_rgb(
#    arr_st,
#    ax=ax,
#    str_clip=0.2,
#    title="composição NIR+R+G",
#)

# Show the image(s)
#plt.show()

# Writing the geotif file

# Create the 3-band raster file

print(arr_st.shape)
#  1
"""
dst_ds = gdal.GetDriverByName('GTiff').Create('myGeoTIFF.tif', nx, ny, 3, gdal.GDT_Int16)
dst_ds.SetGeoTransform(geo)    # specify coords
srs = osr.SpatialReference()            # establish encoding
srs.ImportFromEPSG(32722)                # WGS84 lat/long
dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
dst_ds.WriteArray(arr_st)   # write r-band to the raster
dst_ds.FlushCache()                     # write to disk
dst_ds = None
"""





