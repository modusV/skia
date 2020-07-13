import sys
import pandas as pd
sys.path.append('../src/')
from classification.utils import *
from datetime import datetime
from call import perform_classification


# Insert data information

day = 22
month = 1
year = 2020
segmented = False

filenames = {
    'solpos' :          './exdata/solpos.csv',
    'car_heading':      './exdata/cars_heading.csv',
    'weather':          './exdata/weather.csv',
    'dl_img_path' :     './exdata/segmented/', 
    'dl_img_suffix' :   '_seg_p1.jpg',
    'norm_img_path' :   './exdata/images/',
    'norm_img_suffix' : '_p1.jpg',
}

# Load the datasets and select rows containing data about the needed day of year

solpos = pd.read_csv(filenames['solpos'], index_col=0)
solpos = solpos[(solpos['Day'] == day) & (solpos['Month'] == month)]# & (solpos['Zenith (refracted)'] <= 90)]
cars_heading = pd.read_csv(filenames['car_heading'])
weather = pd.read_csv(filenames['weather'])
weather = weather[(weather['Day'] == day) & (weather['Month'] == month)]
sza_saa = solpos[['Zenith (refracted)', 'Azimuth angle']]


# Merge weather forecast and GHI values with the solar geometry database

tmp = weather.set_index(pd.DatetimeIndex(pd.to_datetime(weather['date_time']))).drop(columns='date_time')
df = pd.DataFrame(columns=tmp.columns, index=sza_saa.index)
df = df.set_index(df.reset_index()['index'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').replace(year=2020)))
merged = pd.merge(left=df, right=tmp, how='left', left_index=True, right_index=True, suffixes=('_y', ''))
merged.drop(merged.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

# Interpolate missing values, needed to go from hourly resolution to 10 minutes one.

fvi = merged.first_valid_index()
lvi = merged.last_valid_index()
merged.loc[fvi:lvi] = merged.loc[fvi:lvi].interpolate().round(3)

# Perform classification

data = perform_classification(day, month, year, cars_heading, filenames, sza_saa, merged, method='original')
data.to_csv('output_prediction.csv')