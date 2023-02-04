"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# -------- Library --------
# Directory manage
import os
from glob import glob

# Data processing
import math
import statistics
from scipy import stats
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import imgaug.augmenters as iaa

# Morphology and Filtering
from skimage.measure import label, regionprops
from skimage.color import label2rgb

# Graphics
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.express as px

# Deep Learning
import tensorflow as tf

# Py import
import tools
import ai_ContexClassify
import filter


# V5
def main(path_dataset, filter_mask: str, classify_mask: False, address_to_save: str):

    # Load the cerradata
    dataset, file_address = tools.load_image(path=path_dataset)
    item = 0

    # Loop 1: Read each cerradata's patch
    for rgba_patch in dataset:
        # Lists
        coord = []
        coord_small = []
        wh_dim = []
        centroid = []
        mask_patch = []

        # New Patch
        classified_mask = np.ones((256, 256))
        classified_mask = Image.fromarray(classified_mask)

        # Masks Generating
        if filter_mask == 'bnow_otsu':
            mask_patch = filter.bnow_otsu(rgba_patch=rgba_patch, classify_it_b4=True)
        elif filter_mask == 'cfps_otsu':
            mask_patch = filter.cfps_otsu(rgba_patch=rgba_patch, classify_it_b4=True)
        else:
            print('Filter not found. Only available: cfps_otsu and bnow_otsu.')

        # When true
        if classify_mask:
            print('--------------------')
            print('AI online.')

            # Get the coordinates of the polygons
            for mascara in mask_patch:
                for region_mask in regionprops(mascara):
                    # It is considered only those polygons which area bigger then 900
                    if region_mask.area > 900:
                        coord.append(region_mask.coords)
                        wh_dim.append(region_mask.bbox)
                        centroid.append(region_mask.centroid)
                    else:
                        coord_small.append(region_mask.coords)

            # Loop 2: Take the number of labels
            for id_label in range(len(coord)):
                print('Polygon: ', id_label)

                # Getting width and height
                height = wh_dim[id_label][2]
                width = wh_dim[id_label][3] - wh_dim[id_label][1]
                print('Width:', str(width), 'Height:', str(height))

                # Temp image: use only to separate the polygon
                polygon_raster = Image.new('RGBA', (256, 256))  # Real Size 256x256

                # Loop 3: Take the number of coordinates
                for id_coord in range(len(coord[id_label])):
                    # [Label] / [rows,column]
                    x, y = list(coord[id_label][id_coord])

                    # Get pixels from patch using the coordinates masks
                    image_polygon = rgba_patch.getpixel((y, x))

                    # Pixels values are give to the temporally polygon_raster
                    polygon_raster.putpixel((y, x), image_polygon)

                """
                plt.figure(figsize=(5, 5))
                plt.imshow(polygon_raster)
                plt.title('Polygon')
                #plt.show()
                """

                # Crop center
                # Get dimensions of the Mask's polygon
                minr, minc, maxr, maxc = wh_dim[id_label]

                # Polygon Crop in the image
                cropped_poly = polygon_raster.crop((minc, minr, maxc, maxr))

                # Crop polygons in patches: each 3 steps, square window of 35%,
                # considering the width and height of the polygon
                croppedPoly = tools.clipping_window_over_polygon(data=cropped_poly, step_size=3,
                                                                 window_size=(int(height * 35/100), int(width * 35/100)))

                # List
                predicts = []

                # Get 20% of patches of the polygons to classify
                qtdMaxPatches = int(len(croppedPoly) * (10 / 100))

                # Define only 80 patches if bigger than
                if qtdMaxPatches > 80:
                    qtdMaxPatches = 80
                    print(str(qtdMaxPatches), 'Patches were selected to classify.')
                else:
                    qtdMaxPatches = qtdMaxPatches
                    print(str(qtdMaxPatches), 'Patches were selected to classify.')

                # Loop 4: Polygon's patches reading
                for patch in croppedPoly[:qtdMaxPatches]:
                    # Array to image
                    patchOfPoly = Image.fromarray(patch)

                    # Fill a new image with cropped polygon
                    fillPolygon = tools.fillaimage(patchOfPoly)

                    # Data normalization 0-1
                    normalized_polygon = tools.normalize_raster(fillPolygon)

                    # Classify: CerraNetv3
                    y_pred = ai_ContexClassify.cerranet_module(data=normalized_polygon)

                    # Save the preds
                    predicts.append(y_pred)

                # Verify amount of patches
                if qtdMaxPatches < 1:
                    print('1 Patch was selected to classify.')
                    # Polygon center crop
                    centerCrop = iaa.CropToFixedSize(height=int(height * 50/100), width=int(width * 50/100))
                    polygonCropC = centerCrop.augment_image(np.array(cropped_poly))

                    # Array to image
                    patchOfPoly = Image.fromarray(polygonCropC)

                    # Fill a new image with cropped polygon
                    fillPolygon = tools.fillaimage(patchOfPoly)

                    # Data normalization 0-1
                    normalized_polygon = tools.normalize_raster(fillPolygon)

                    # Classify: CerraNetv3
                    id_pred = ai_ContexClassify.cerranet_module(data=normalized_polygon)

                else:
                    # Calculate Mode
                    moda_pred = stats.mode(np.array(predicts))
                    print(moda_pred)
                    id_pred = int(moda_pred[0][0][0])

                print('Moda da predição:', id_pred)

                # loop 4: Replaces polygon's pixels for model classified pixels
                for id_coord in range(len(coord[id_label])):
                    #  Mask's [rows][column] of the Polygon
                    x, y = list(coord[id_label][id_coord])

                    # Pixels are gave to the new image
                    classified_mask.putpixel((y, x), id_pred)

                # Loop 5: Replaces polygon's pixels, which area is less than 900
                for id_slabel in range(len(coord_small)):
                    # Loop 6: Replace those small polygons (poly <900) by neighborhood pixels
                    for id_scoord in range(len(coord_small[id_slabel])):
                        #  Mask's [rows][column] of the Polygon
                        x, y = list(coord_small[id_slabel][id_scoord])
                        # Pixels are gave to the new image
                        classified_mask.putpixel((y, x), id_pred)

            print('Saving the unlabeled Masks.')
            # Save the unlabeled mask
            tools.save_mask(masks=classified_mask, file_address=file_address[item],
                            address_to_save=address_to_save, classify_mask=classify_mask)
            item = item + 1


            plt.figure(figsize=(10, 10))
            print('Classes do polygon: ', np.unique(tf.keras.utils.img_to_array(classified_mask)))
            plt.imshow(tf.keras.utils.img_to_array(classified_mask), cmap='CMRmap')
            plt.title('Classified Polygon')
            plt.show()

            del coord
            del coord_small
            del wh_dim
            del centroid
            del mask_patch
        # When False
        else:
            print('Saving the unlabeled Masks.')
            # Save the unlabeled mask
            tools.save_mask(masks=mask_patch, file_address=file_address[item],
                            address_to_save=address_to_save, classify_mask=classify_mask)
            item = item + 1


path_input = '../data/cerradata/test/patches/building/20200428_209_133_L4_14597*.tif'
path_output = '../data/cerradata/test'
if __name__ == '__main__':
    main(path_dataset=path_input, filter_mask='cfps_otsu', classify_mask=True, address_to_save=path_output)
