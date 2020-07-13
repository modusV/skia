
import math
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps
from pvlib.irradiance import erbs
from skimage.filters import threshold_otsu


def pano_to_fisheye(im):
    """
    Description of pano_to_fisheye
    Transforms an image from cylindrical view to azimuthal view as described in:
        - http://www.sciencedirect.com/science/article/pii/S0169204617301950

    Args:
        im (undefined): PIL Image

    """
    pano_height = im.size[1]
    pano_width = im.size[0]

    fisheye_image = Image.new('RGB', (pano_width, pano_height))
    fisheye_imageSZA = Image.new('F', (pano_width, pano_height))
    fisheye_imageSAA = Image.new('F', (pano_width, pano_height))

    Wc = pano_width
    Hc = pano_height/2
    r0 = math.floor(Wc/(2*np.pi))
    Cx = r0
    Cy = r0

    for i in range(0, math.floor(Wc/np.pi)):
        for j in range(0, math.floor(Wc/np.pi)):
            r = np.sqrt((i-Cx)**2 + (j-Cy)**2)
            if r > r0 or r == 0:
                fisheye_image.putpixel((i, j), (0,0,0))
                fisheye_imageSZA.putpixel((i, j), np.nan)
                fisheye_imageSAA.putpixel((i, j), np.nan)
                continue
            if i < Cx:
                theta = 3*np.pi/2 - np.arctan((j - Cy)/(i - Cx))
            elif i > Cx:
                theta = np.pi/2 - np.arctan((j - Cy)/(i - Cx))
            elif i == Cx and j > Cy:
                theta = 2* np.pi
            elif i == Cx and j < Cy:
                theta = np.pi
            Xc = math.floor(theta * Wc/(2*np.pi))-1
            Yc = math.floor((r*Hc)/r0)-1
            px = im.getpixel((Xc, Yc))
            
            fisheye_image.putpixel((i, j), px)
            fisheye_imageSZA.putpixel((i, j), (r/r0) * 90)
            fisheye_imageSAA.putpixel((i, j), (theta * 180)/np.pi)
    
    fisheye_image = ImageOps.flip(fisheye_image.crop((0, 0, math.floor(Wc/np.pi), math.floor(Wc/np.pi))).rotate(180))
    fisheye_imageSZA = ImageOps.flip(fisheye_imageSZA.crop((0, 0, math.floor(Wc/np.pi), math.floor(Wc/np.pi)))).rotate(270)
    fisheye_imageSAA = ImageOps.flip(fisheye_imageSAA.crop((0, 0, math.floor(Wc/np.pi), math.floor(Wc/np.pi))).rotate(90))
    return fisheye_image, fisheye_imageSZA, fisheye_imageSAA



def orient_pano(pano, heading):
    """
    Description of orient_pano
    
    orients a panoramic image to face North

    Args:
        pano (undefined): PIL Image
        heading (undefined): Degrees from North

    """
    if heading <= 180:
        n_pix = round((heading/360) * pano.width)
        to_move = Image.fromarray(np.asarray(pano)[:, pano.width-n_pix:])
        rest = Image.fromarray(np.asarray(pano)[:, :pano.width-n_pix])
        dst = Image.new('RGB', (pano.width, pano.height))
        dst.paste(to_move, (0, 0))
        dst.paste(rest, (to_move.width, 0))
        return dst
    else:
        n_pix = round(((360-heading)/360) * pano.width)
        to_move = Image.fromarray(np.asarray(pano)[:, :n_pix])
        rest = Image.fromarray(np.asarray(pano)[:, n_pix:])
        dst = Image.new('RGB', (pano.width, pano.height))
        dst.paste(rest, (0, 0))
        dst.paste(to_move, (rest.width, 0))
        return dst



def get_position_photographic(SZA, SAA, fishSZA, fishSAA):
    """
    Description of get_position_photographic
    
    Retrieves position in the azimuthal fisheye image using solar zenith angle and solar azimuth angle

    Args:
        SZA (undefined): Solar Zenith Angle in degrees
        SAA (undefined): Solar Azimuth Angle in degrees
        fishSZA (undefined): Map of Solar Zenith Angles
        fishSAA (undefined): Map of Solar Azimuth Angles

    """
    tdata = (np.asarray(fishSZA) - SZA)**2 + (np.asarray(fishSAA) - SAA)**2
    pos = np.unravel_index(np.nanargmin(tdata.ravel()), tdata.shape)
    return int(pos[0]), int(pos[1])



def trace_sun_position(image, sun_position, fish_SZA, fish_SAA):
    """
    Description of trace_sun_position
    
    Traces sun trajectory on the azimuthal fisheye image using sun position.

    Args:
        image (undefined): PIL image
        sun_position (undefined): array containing solar zenith angle and solar azimuth angle.
        fish_SZA (undefined): Map of Solar Zenith Angles
        fish_SAA (undefined): Map of Solar Azimuth Angles

    """
    image = image.copy().rotate(180)
    for SZA, SAA in sun_position:
        if SZA < 90:
            #position = get_position(SZA, SAA, 150)
            position = get_position_photographic(SZA, SAA, fish_SZA, fish_SAA)
            image.putpixel(position, (255,0,0))
    return image.rotate(180)


def get_obstruction(class_im, sun_data, fishSZA, fishSAA):
    """
    Description of get_obstruction
    
    For each provided sun position, returns whether the sun is obstructed at that position or not.

    Args:
        class_im (undefined): PIL Image
        sun_data (undefined): array containing solar zenith angle and solar azimuth angle.
        fishSZA (undefined): Map of Solar Zenith Angles
        fishSAA (undefined): Map of Solar Azimuth Angles

    """
    obstruction = []
    class_im = class_im.copy().rotate(180)
    for SZA, SAA in sun_data:
        if SZA >= 90:
            obstruction.append(np.nan)
        else:
            position = get_position_photographic(SZA, SAA, fishSZA, fishSAA)
            px = class_im.getpixel(position)
            if px == (255, 255, 255):
                obstruction.append(0)
            elif px == (0, 255, 0):
                obstruction.append(1)
            else:
                obstruction.append(2)
    return pd.DataFrame(dict(Obsc=obstruction))


def ghi_from_obstruction(GHI, sza, obstr, SVF, doy):
    """
    Description of ghi_from_obstruction
    
    Returns solar irradiance value for a location

    Args:
        GHI (undefined): Global Horizontal Irradiance
        sza (undefined): Solar Zenith Angle in degrees
        obstr (undefined): Obstruction information calculated with get_obstruction method
        SVF (undefined): Sky View Factor
        doy (undefined): Day of the year

    """
    sol = erbs(GHI, sza, doy)
    if obstr != 0:
        obstr = 0
    else:
        obstr = 1
    return sol['dni'] * np.cos(np.deg2rad(sza)) * obstr + SVF * sol['dhi']




def count_img_pixels(r):
    """
    Description of count_img_pixels
    Count number of pixel in the circle, using 
        - https://math.stackexchange.com/questions/888572/finding-number-of-pixels-in-a-circle-using-diameter

    Args:
        r (undefined): radius of image

    """
    pixel_circle = np.round(1 + 4 * sum([abs((r**2) / (4*i +1)) - abs((r**2) / (4*i+3)) for i in range(0, 20000)])).astype(int)
    #N = 1 + 4 * sum([1+0.5*np.sqrt(301**2-4*y**2) for y in range(0,150)])
    pixel_total = 301*301
    #N_area = round(np.pi * (150**2))
    return pixel_circle, pixel_total



def find_otsu_tresh(image):
    """
    Description of find_otsu_tresh
    Finds colour treshold using Otsu's method for buildings and trees on an image

    Args:
        image (undefined): PIL Image

    """
    arr_img = np.asarray(image)
    green_vals = []
    blu_vals = [brightness_calc(px) for row in arr_img for px in row]
    blu_tresh = threshold_otsu(np.asarray([a for a in blu_vals if a != 0]))


    for i,row in enumerate(arr_img):
        t = []
        g = []
        for j,px in enumerate(row):
            brightness = brightness_calc(px)
            green = green_calc(px)

            if brightness < blu_tresh:
                g.append(green)

        green_vals.append(g)

    values = np.asarray([val for row in green_vals for val in row if val != 0])
    
    # Check if there is another colour apart from blue
    if not np.all(values == values[0]):
        green_tresh = threshold_otsu(values)
        return blu_tresh, green_tresh
    else:
        return blu_tresh, 260



def get_SVF(image):
    """
    Description of get_SVF
    Return the percentage of the sky visible from that location, it is a number between one and zero

    Args:
        image (undefined): PIL image

    """
    clearsky = (255,255,255)
    resh_im = np.asarray(image).reshape(90601, 3)
    return sum([(r,g,b) == clearsky for r,g,b in resh_im])/round((image.width/2)**2 * np.pi)
    

def is_inside_image(xp, yp, r):
    """
    Description of is_inside_image
    Returns True if the point is inside the image, False otherwise

    Args:
        xp (undefined): x pixel
        yp (undefined): y pixel
        r (undefined): radius of image

    """
    return ((xp - r)**2 + (yp - r)**2) <= r**2


def ghi_to_w(ghi, eff=0.19, surf=0.0568):
    """
    Description of ghi_to_w

    Args:
        ghi (undefined): Global Horizontal Irradiance
        eff=0.19 (undefined): Efficiency of solar panel
        surf=0.0568 (undefined): Solar panel surface area

    """
    return ghi*eff*surf


def ma_to_w(ma, volt):
    """
    Description of ma_to_w
    
    Transforms milliampere and voltage into watts

    Args:
        ma (undefined): milliampere
        volt (undefined): voltage

    """
    return (ma/1000) * volt

def brightness_calc(pixel):
    """
    Description of brightness_calc
    
    calculates adjusted brightness level of a pixel emphatising blue colour

    Args:
        pixel (undefined): pixel in RGB format

    """
    return (0.5 * pixel[0] + pixel[1] + 1.5 * pixel[2])/3


def green_calc(pixel):
    """
    Description of green_calc

    calcualtes adjusted brightness of pixel emphatising green colour

    Args:
        pixel (undefined): pizel in RGB format

    """
    return (2 * pixel[1] - pixel[0] - pixel[2])/3
