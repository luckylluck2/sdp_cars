import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

# Load vehicles data

clean_data_folder = './data'
vehicles_file = 'vehicles.parquet'
cleaned_file = 'cleaned_data.parquet'

data = pd.read_parquet(os.path.join(clean_data_folder, vehicles_file))

# all_columns = ['id', 'url', 'region', 'region_url', 'price', 
#                'year', 'manufacturer', 'model', 'condition', 
#                'cylinders', 'fuel', 'odometer', 'title_status', 
#                'transmission', 'VIN', 'drive', 'size', 'type', 
#                'paint_color', 'image_url', 'description', 'county', 
#                'state', 'lat', 'long', 'posting_date']
categorical_columns = ['manufacturer', 'condition', 'fuel', 'title_status', 'transmission',
                       'drive', 'size', 'type', 'paint_color', 'cylinders']
non_categorical_columns = ['price', 'year', 'odometer', 'lat', 'long']
required_columns = categorical_columns + non_categorical_columns

#select only the wanted columns
data = data[required_columns]

column_filters = {'price': {'min': 1000, 'max': 50000, 'exclude': [], 'impute': 'median'}, 
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
    print(f'Filtered {column}, removed {old_length - len(data)} rows')
    return new_data

print('Original number of rows:', len(data))

for column in column_filters:
    
    data = take_subset(data, column, 
                       min_val=column_filters[column]['min'],
                       max_val=column_filters[column]['max'],
                       exclude_vals=column_filters[column]['exclude'],
                       impute=column_filters[column]['impute'])
    

print('Remaining number of rows:', len(data))

#ONE HOT ENCODING! WHOOP WHOOP!
enc = OneHotEncoder(sparse_output=False).fit(data[categorical_columns])
encoded = enc.transform(data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out())

#indexing went 'wrong' somewhere
data.index = encoded_df.index

data = pd.concat([data[non_categorical_columns].copy(), encoded_df], axis = 1)
data = data.drop([cat_column + '_None' for cat_column in categorical_columns], axis=1)

print('Number of columns:', len(data.columns.to_list()))
print('Number of NaNs:')
print(data.isna().sum().to_string())

data.to_parquet(os.path.join(clean_data_folder, cleaned_file))
