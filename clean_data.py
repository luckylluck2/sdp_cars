import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

# Load vehicles data

clean_data_folder = './data'
vehicles_file = 'vehicles.parquet'
cleaned_file = 'cleaned_data.parquet'
cleaned_file_categorical = 'cleaned_data_categorical.parquet'

odometer_cutoffPoint = 400000   #in miles
price_cutoffPoint = 500000      #in USD

required_columns = ['price', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer',
                    'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'lat', 'long']

og_data = pd.read_parquet(os.path.join(clean_data_folder, vehicles_file))

column_filters = {'price': {'min': 500, 'max': 70000, 'exclude': [], 'impute': 'median'}, 
                  'year': {'min': 2000, 'max': 2020, 'exclude': [], 'impute': 'median'}, 
                  'manufacturer': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'condition': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'cylinders': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'fuel': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'odometer': {'min': 0, 'max': 300000, 'exclude': [], 'impute': 'median'},
                  'title_status': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'transmission': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'drive': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'size': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'type': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'paint_color': {'min': None, 'max': None, 'exclude': [], 'impute': None},
                  'lat': {'min': None, 'max': None, 'exclude': [], 'impute': 'median'},
                  'long': {'min': None, 'max': None, 'exclude': [], 'impute': 'median'}}

def take_subset(data, column, min_val=None, max_val=None, exclude_vals=None, impute=None):
    new_data = data.copy()
    old_length = len(new_data)
    if impute is not None:
        if impute == 'median':
            new_data[column] = new_data[column].fillna(new_data[column].median(skipna=True))
    if min_val is not None:
        new_data = new_data[new_data[column] >= min_val]
    if max_val is not None:
        new_data = new_data[new_data[column] <= max_val]
    if exclude_vals is not None:
        if type(exclude_vals) == list:
            for val in exclude_vals:
                new_data = new_data[new_data[column] != val]
        else:
            new_data = new_data[new_data[column] != exclude_vals]
    print(f'Filtered {column}, removed {old_length - len(new_data)} rows')
    return new_data


for column in column_filters:
    
    og_data = take_subset(og_data, column, 
                       min_val=column_filters[column]['min'],
                       max_val=column_filters[column]['max'],
                       exclude_vals=column_filters[column]['exclude'],
                       impute=column_filters[column]['impute'])


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


#select only the wanted columns
clean_data = og_data[required_columns]

#ONE HOT ENCODING! WHOOP WHOOP!
categorical_columns = ['manufacturer', 'condition', 'fuel', 'title_status', 'transmission',
                       'drive', 'size', 'type', 'paint_color'] 
enc = OneHotEncoder(sparse_output=False).fit(clean_data[categorical_columns])
encoded = enc.transform(clean_data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out()) 
# encoded_df.columns.to_list()

non_categorical_columns = ['price', 'year', 'odometer', 'cylinders']

#indexing went 'wrong' somewhere
clean_data.index = encoded_df.index

clean_data = pd.concat([clean_data[non_categorical_columns].copy(), encoded_df], axis = 1)

og_data[required_columns].to_parquet(os.path.join(clean_data_folder, cleaned_file_categorical))

clean_data.to_parquet(os.path.join(clean_data_folder, cleaned_file))

# clean_data.to_csv(os.path.join(clean_data_folder,  "clean_data.csv"))

print(len(clean_data))