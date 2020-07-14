# skia
Calculating solar irradiance in street canyons using Google Street View (GSV) images.

## Needed data

1. File containing GSV cars panorama ids and car heading in degrees with respect to north. It can be downloaded using the tool at https://github.com/modusV/gsvpypano
2. Images correponding to those points; a tool to retrieve these panoramas is available in the just mentioned repository, **gsvpypano**. If the brightness based method is chosen, use `full=True` in the download tool and provide the images 'as they are'. If the deep learning method is preferred, use `cropped=True`, classify them using the Pyramid Scene Parsing network found at https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow, and recompose the output frames with the `compose_folder()` method found in **gsvpypano**.
3. Dataframe with the sun location in the sky with a 10 minutes resolution. It is available for download at https://midcdmz.nrel.gov/solpos/solpos.html. The most important fields to check are 'Solar Zenith Angle' and 'Solar Azimuth Angle'.
4. Weather data of the location with a column indicating the Global Horizontal Irradiance for that specific time. These data are available at several websites, such as https://www.worldweatheronline.com/developer/. The GHI values can be obtained using the model presented at the repository https://github.com/modusV/fos


## Usage
Install the requirements using ```pip install -r requirements.txt```

The ``call.py`` file contains an high-level interface to the library, which allows to obtain the final prediction at once. It accepts as input parameters:

- Day
- Month
- Year
- cars_heading, a pandas DataFrame containing the information about Ids (which corresponds to the image names, if the provided tool is used to retrieve these data) and Heading direction.
- filenames, a dictionary containing the images paths and suffixes in this format:

```python
filenames = {
    'dl_img_path' :     './exdata/segmented/', 
    'dl_img_suffix' :   '_seg_p1.jpg',
    'norm_img_path' :   './exdata/images/',
    'norm_img_suffix' : '_p1.jpg',
}
```
- solpos, pandas DataFrame with two columns, one representing the solar azimuth angle and the other containing the solar zenith angle 
- weather, dataframe of the weather forecast plus a column containing the Global Horizontal Irradiance value for that precise conditions
- method, can be 'segmented' if the output of the PSPnet is used or 'brightness' if the original images are chosen. The segmented method has a better accuracy but the brightness one is faster because the classification step is skipped.



## Aknowledgments
I would like to personally mention Fang-Ying Gong, because the code implemented for transforming the images from cylindrical to fisheye, to orient panoramas to north and to find a pixel on the fish-eye image has been translated from her original matlab code.

## References

- The 'segmented method' is described at: http://www.sciencedirect.com/science/article/pii/S0360132318306437
- Brightness method is presented at: http://www.sciencedirect.com/science/article/pii/S0169204617301950
