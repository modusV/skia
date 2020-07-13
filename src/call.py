from datetime import datetime
from tqdm import tqdm
from PIL import Image
from classification.utils import *
from classification.classify import *


def perform_classification(day, month, year, cars_heading, filenames, solpos, weather, method='segmented'):
    """
    Description of perform_classification
    
    Performs the classification of a series of input points calculating the GHI at ground level 
    for a set of given GSV points during a given day with a ten minutes resolution.
    
    It is possible to do the classification using a brightness based approach or a deep learning approach.
    
    Returns a dataset containing, for each given point, the Irradiance during the specified day.

    Args:
        day (undefined): day of the month
        month (undefined): month
        year (undefined): year
        cars_heading (undefined): Dataset containing GSV cars directions
        filenames (undefined): Filenames of the images 
        solpos (undefined): Dataset containing Solar position information
        weather (undefined): Dataset containing weather forecast associated with Global Horizontal Irradiance
        method='segmented' (undefined): Method to classify sky, buildings and trees specifying type of input images; can be 'segmented' or 'original'

    """
    
    doy = datetime(day=22, month=1, year=2020).timetuple().tm_yday
    #weather = weather[(weather['Day'] == day) & (weather['Month'] == month)]


    for car_idx, car_row in tqdm(cars_heading.iterrows(), total=cars_heading.shape[0]):
        
        if method == 'segmented':
            pano = Image.open(filenames['dl_img_path'] + str(car_row['NumId']) + filenames['dl_img_suffix'])
        else:
            pano = Image.open(filenames['norm_img_path'] + str(car_row['NumId']) + filenames['norm_img_suffix'])
        heading = car_row['Heading']
        oriented_pano = orient_pano(pano, heading)
        fish_pano, fish_SZA, fish_SAA = pano_to_fisheye(oriented_pano)
        ms_image = meanshift_image(fish_pano)

        if method == 'segmented':
            classified, SVF, TVF = brightness_classification(ms_image)
        else:
            classified = adj_nn_colors(ms_image)
            SVF = get_SVF(classified)
        
        obstruction = get_obstruction(classified, solpos.values, fish_SZA, fish_SAA)
        weather['Obsc'] = obstruction.set_index(weather.index)

        ghi_obs = []
        for row_idx, row in weather.iterrows():
            if np.isnan(row['GHI']):
                ghi_obs.append(np.nan)
            else:
                ghi_obs.append(ghi_from_obstruction(row['GHI'], row['Solar Zenith Angle'], row['Obsc'], SVF, doy))
        weather['ghi_' + str(car_row['NumId'])] = ghi_obs
        
        return weather.drop(columns=['Obsc'])

