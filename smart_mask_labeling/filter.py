"""
Software Open Source
©Copyright Mateus de Souza Miranda, 2023
mateus.miranda@inpe.br; mateusmirandaa2@hotmail.com
National Institute for Space Research (INPE)
São José dos Campos, São Paulo, Brazil

Version 4:  January, 2023
"""

# -------- Library --------
# Data
import numpy as np
from PIL import Image

# Morphology and Filtering
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, erosion, opening, diameter_closing
from skimage.morphology import square, convex_hull_image, convex_hull_object, disk
from skimage.color import label2rgb

# Py Import
import ai_ContexClassify
import tools


# -------- Filters --------
# Classes: Cultivated area, Forest, Pasture, Savanna
def cfps_otsu(rgba_patch, classify_it_b4: False):
    # List
    masks = []

    # Default settings
    plus_thresh = 0.01       # Adjust the threshold: the higher it is, the more the pixels connect
    square_open = 10         # Area of opening
    square_dilation = 8      # Area of dilation

    # When True
    if classify_it_b4:
        print('AI Module online.')

        # Data normalization 0-1
        normalized_polygon = tools.normalize_raster(rgba_patch)

        id_class = ai_ContexClassify.cerranet_module(normalized_polygon)

        if id_class == 1:
            # CA
            plus_thresh = 0.01
            square_open = 8
            square_dilation = 8
        elif id_class == 2:
            # FF
            plus_thresh = 0.01
            square_open = 14
            square_dilation = 6
        elif id_class == 5:
            # PA
            plus_thresh = 0.01
            square_open = 10
            square_dilation = 16
        elif id_class == 6:
            # SA
            plus_thresh = 0.01
            square_open = 10
            square_dilation = 5
        else:
            # Keeps the default settings
            plus_thresh = 0.01
            square_open = 10
            square_dilation = 8

    # When False
    else:
        # Keeps the default settings
        plus_thresh = 0.01
        square_open = 10
        square_dilation = 8

    # RGBA to Gray scale
    gray_patch = tools.rgb_to_gray(rgba_patch)
    # Filter
    otsu = threshold_otsu(gray_patch) + plus_thresh

    # morphology
    fd = opening(gray_patch, square(square_open))
    dil = dilation(fd, square(square_dilation))

    # Thresh 1
    thresh1 = dil < otsu
    # Label image region
    mask1 = label(thresh1)
    # Save all masks created
    masks.append(mask1)

    # Thresh 2
    thresh2 = dil > otsu
    # Label image region
    mask2 = label(thresh2)
    # Save all masks created
    masks.append(mask2)
    # Return a list of masks
    return masks


# Classes: Building, Non observed area, Other uses, and Water
def bnow_otsu(rgba_patch, classify_it_b4: False):
    # List
    masks = []

    # Default Settings
    plus_thresh = 0.01          # Adjust the threshold: the higher it is, the more the pixels connect
    square_dilation = [5, 14]   # Area of dilation
    square_erosion = [2, 10]    # Area of erosion          # Area of erosion
    diameter_dt = 20            # diameter_threshold
    diameter_ct = 20            # connectivity

    # When true
    if classify_it_b4:
        print('AI Module online.')
        # Data normalization 0-1
        normalized_polygon = tools.normalize_raster(rgba_patch)
        # AI module: classify before of mask creating
        id_class = ai_ContexClassify.cerranet_module(normalized_polygon)

        if id_class == 0:
            # Building
            plus_thresh = 0.01
            square_dilation = [7, 4]
            square_erosion = [3, 8]
            diameter_dt = 100
            diameter_ct = 100
        elif id_class == 3:     # NO
            # Non observed
            plus_thresh = 0.01
            square_dilation = [8, 2]
            square_erosion = [3, 10]
            diameter_dt = 8
            diameter_ct = 8
        elif id_class == 4:     # OU
            # Other uses
            plus_thresh = 0.01
            square_dilation = [8, 2]
            square_erosion = [3, 10]
            diameter_dt = 8
            diameter_ct = 8
        elif id_class == 7:     # WT
            # Water
            plus_thresh = 0.01
            square_dilation = [8, 12]
            square_erosion = [3, 6]
            diameter_dt = 8
            diameter_ct = 8
        else:
            # Default Settings
            plus_thresh = 0.01  # Adjust the threshold: the higher it is, the more the pixels connect
            square_dilation = [5, 14]  # Area of dilation
            square_erosion = [2, 10]  # Area of erosion          # Area of erosion
            diameter_dt = 20  # diameter_threshold
            diameter_ct = 20  # connectivity
    # When False
    else:
        # Keeps the default settings
        plus_thresh = 0.01
        square_dilation = [5, 14]
        square_erosion = [2, 10]
        diameter_dt = 20
        diameter_ct = 20

    # RGBA to Gray scale
    gray_patch = tools.rgb_to_gray(rgba_patch)

    # Filter
    # Filter Otsu
    otsu = threshold_otsu(gray_patch) + plus_thresh

    # For light regions

    # Morphologies
    dil = dilation(gray_patch, square(square_dilation[0]))
    ero = erosion(gray_patch, square(square_erosion[0]))
    dcl = diameter_closing(dil, diameter_threshold=diameter_dt, connectivity=diameter_ct)

    # Combining morphologies
    morphology = (dil - ero + dcl)

    # Threshold 1: lightest regions: Building, other used, bare soil
    thresh1 = morphology > otsu
    # Label image region
    mask1 = label(thresh1)
    masks.append(mask1)

    # For darkest regions: areas with vegetation or other

    # Morphologies
    ero2 = erosion(gray_patch, square(square_erosion[1]))
    dil2 = dilation(ero2, square(square_dilation[1]))
    dcl2 = diameter_closing(gray_patch, diameter_threshold=diameter_dt, connectivity=diameter_ct)

    # Combining morphologies
    morphology2 = (dil2 - ero2 + dcl2)

    thresh2 = morphology2 < otsu
    # Label image region
    mask2 = label(thresh2)
    masks.append(mask2)
    # Return a list of masks
    return masks
