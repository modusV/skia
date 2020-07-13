
import numpy as np
from PIL import Image
import pymeanshift as pms
from scipy.spatial.distance import cdist
from skimage.filters import threshold_otsu
from classification.utils import *


def meanshift_image(image):
    """
    Description of meanshift_image
    
    Performs meanshift algorithm on the image

    Args:
        image (undefined): PIL image

    """
    
    (segmented_image, labels_image, number_regions) = pms.segment(image, spatial_radius=6, range_radius=5, min_density=100)
    return Image.fromarray(segmented_image)


def adj_nn_colors(image):
    """
    Description of adj_nn_colors
    
    Adjusts all the colors of an image to a pre-defined palette.

    Args:
        image (undefined): PIL image

    """
    palette = np.array([
        [108, 225, 228], #sky, acquamarine blu
        [169, 123, 120], #buildings, pharlaph
        [94, 194, 57], #trees, mantis
        [138, 141, 143], #sea, misclassified
        [0, 0, 0], #bg

        [225, 62, 235], #buses, fuchsia
        [48, 102, 191], #cars, blue
        [245, 251, 79], #sidewalks, yellow
        [140, 140, 140], #street, grey
        [254, 0, 26], #something, red

    ])


    palette_dest = np.array([
        [255, 255, 255], #sky
        [128, 128, 128], #building
        [0, 255, 0], #trees
        [255, 255, 255], #sky
        [0, 0, 0], #bg

        [128, 128, 128], #building
        [128, 128, 128], #building
        [128, 128, 128], #building
        [128, 128, 128], #building
        [128, 128, 128], #building
    ])

    old_img = np.array(image)
    new_image = []

    for arr in old_img:
        closest_idx = cdist(arr, palette).argmin(1)
        data_in_palette = palette_dest[closest_idx]
        new_image.append(data_in_palette)

    arr_image = np.asarray(new_image, dtype=np.uint8)
    return Image.fromarray(arr_image)



#LI method for segmentation

def brightness_classification(ms_image):
    """
    Description of brightness_classification
    
    Classifies pixels of image using a simplified version of the method described in:
        - http://www.sciencedirect.com/science/article/pii/S0169204617301950
    Geometrical rules are not applied.

    Args:
        ms_image (undefined): PIL Image

    """
    
    blu_tresh, green_tresh = find_otsu_tresh(ms_image)

    new = []
    fisheye_image_seg = Image.new('RGB', (ms_image.size[0], ms_image.size[1]))

    SVF = 0
    TVF = 0

    pixel_circle, pixel_total = count_img_pixels(ms_image.width/2)
    
    green_values = []
    ms_image_arr = np.asarray(ms_image)

    for i,row in enumerate(ms_image_arr):
        t = []
        g = []
        for j,px in enumerate(row):

            if is_inside_image(i, j, ms_image.width/2):
                brightness = brightness_calc(px)
                green = green_calc(px)
                t.append(brightness)

                if brightness >= blu_tresh:
                    fisheye_image_seg.putpixel((i, j), (255,255,255))
                    SVF += 1
                else:
                    g.append(green)
                    if green <= green_tresh:
                        fisheye_image_seg.putpixel((i, j), (128, 128, 128))
                    else:
                        fisheye_image_seg.putpixel((i, j), (0,255,0))
                        TVF += 1
            else:
                fisheye_image_seg.putpixel((i, j), (0,0,0))

        new.append(t)
        green_values.append(g)

    return ImageOps.flip(fisheye_image_seg.rotate(90)), SVF/pixel_circle, TVF/pixel_circle
