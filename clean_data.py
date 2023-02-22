import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

# Load vehicles data

clean_data_folder = './data'
vehicles_file = 'vehicles.parquet'
cleaned_file = 'cleaned_data.parquet'

odometer_cutoffPoint = 400000   #in miles
price_cutoffPoint = 500000      #in USD

required_columns = ['price', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer',
                    'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 
                    'yearSquared', 'logOdometer']

og_data = pd.read_parquet(os.path.join(clean_data_folder, vehicles_file))

#drop rows with extremely large prices
og_data.drop(np.where(og_data['price'] > price_cutoffPoint)[0], inplace=True)

# Select appropriate columns


def cylinders_to_numeric(cylinderstring):
    """
    Convert # cylinders as a string, to integer value
    """
    if ((cylinderstring == 'other') | (cylinderstring == None)): return None
    else: return(int(cylinderstring.replace(' cylinders', '')))

og_data['cylinders'] = og_data['cylinders'].map(cylinders_to_numeric)
#also, impute any missing values with the mean
og_data['cylinders'] = og_data['cylinders'].fillna(og_data['cylinders'].mean(skipna=True))

#Next, we ignore rows with unrealistic values. Any abnormally high values for specific columns are removed.
#The odometer has some unrealistic values, so any value exceeding a threshold (set above) is set to be missing
og_data['odometer'][og_data['odometer'] > odometer_cutoffPoint] = np.nan
#and we impute missing values with the mean
og_data['odometer'] = og_data['odometer'].fillna(og_data['odometer'].mean(skipna=True))

og_data['odometer'][og_data['odometer'] == 0] = 1

#we also add some additional features:
og_data['yearSquared'] = og_data['year']**2     #to explain behaviour with extremele new/old cars
og_data['logOdometer'] = np.log10(og_data['odometer'])

#impute missing years with mean values:
og_data['year'] = og_data['year'].fillna(og_data['year'].mean(skipna=True))
og_data['yearSquared'] = og_data['yearSquared'].fillna(og_data['yearSquared'].mean(skipna=True))

#select only the wanted columns
clean_data = og_data[required_columns]

#ONE HOT ENCODING! WHOOP WHOOP!
categorical_columns = ['manufacturer', 'condition', 'fuel', 'title_status', 'transmission',
                       'drive', 'size', 'type', 'paint_color'] 
enc = OneHotEncoder(sparse=False).fit(clean_data[categorical_columns])
encoded = enc.transform(clean_data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out()) 
# encoded_df.columns.to_list()

non_categorical_columns = ['price', 'year', 'odometer', 'cylinders', 'logOdometer', 'yearSquared']


clean_data = pd.concat([clean_data[non_categorical_columns].copy(), encoded_df], axis = 1)

#clean_data.dtypes

clean_data.to_parquet(os.path.join(clean_data_folder, cleaned_file))

len(clean_data)