import os
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# Load vehicles data

clean_data_folder = './data'
vehicles_file = 'vehicles.parquet'
cleaned_file_train = 'cleaned_data_train.parquet'
cleaned_file_test = 'cleaned_data_test.parquet'
cleaned_file_val = 'cleaned_data_val.parquet'
cleaned_file_train_price = 'cleaned_price_train.parquet'
cleaned_file_test_price = 'cleaned_price_test.parquet'
cleaned_file_val_price = 'cleaned_price_val.parquet'
cleaned_file_categorical = 'cleaned_data_categorical.parquet'

og_data = pd.read_parquet(os.path.join(clean_data_folder, vehicles_file))
n_og_data = len(og_data)

print('Number of rows in original data:', n_og_data)
print('Columns in original data:', og_data.columns.to_list())
print(' ')

def cylinders_to_numeric(cylinderstring):
    """
    Convert # cylinders as a string, to integer value
    """
    if ((cylinderstring == 'other') | (cylinderstring == None)): return None
    else: return(int(cylinderstring.replace(' cylinders', '')))

og_data['cylinders'] = og_data['cylinders'].map(cylinders_to_numeric)

column_filters = {'price': {'min': 500, 'max': 70000, 'exclude': [], 'impute': None}, 
                  'year': {'min': 2000, 'max': 2020, 'exclude': [], 'impute': 'median'}, 
                  'manufacturer': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'condition': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'cylinders': {'min': None, 'max': None, 'exclude': [], 'impute': 'median'}, 
                  'fuel': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'odometer': {'min': 0, 'max': 300000, 'exclude': [], 'impute': 'median'},
                  'title_status': {'min': None, 'max': None, 'exclude': ['parts only'], 'impute': None}, 
                  'transmission': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'drive': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'size': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'type': {'min': None, 'max': None, 'exclude': [], 'impute': None}, 
                  'paint_color': {'min': None, 'max': None, 'exclude': [], 'impute': None},
                  'lat': {'min': None, 'max': None, 'exclude': [], 'impute': 'median'},
                  'long': {'min': None, 'max': None, 'exclude': [], 'impute': 'median'}}

def take_subset(data, column, min_val=None, max_val=None, exclude_vals=None):
    new_data = data.copy()
    old_length = len(new_data)
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
    n_removed_rows = old_length - len(new_data)
    if n_removed_rows > 0:
        print(f'Filtered {column}, removed {n_removed_rows} rows')
    return new_data

imputed_vals = {}
for column in column_filters:
    if column_filters[column]['impute'] == 'median':
        median = og_data[column].median(skipna=True)
        imputed_vals.update({column:median})
        og_data[column] = og_data[column].fillna(median)
    else:
        imputed_vals.update({column:None})

for column in column_filters:
    og_data = take_subset(og_data, column, 
                       min_val=column_filters[column]['min'],
                       max_val=column_filters[column]['max'],
                       exclude_vals=column_filters[column]['exclude'])
    
data_filters = {}
for column in column_filters:
    col_type = 'categorical' if og_data[column].dtype == object else 'numeric'
    if col_type == 'numeric':
        col_min = float(og_data[column].min(skipna=True))
        col_max = float(og_data[column].max(skipna=True))
        col_mean = float(og_data[column].mean(skipna=True))
        col_median = float(og_data[column].median(skipna=True))
        col_impute_type = column_filters[column]['impute']
        col_impute_value = imputed_vals[column]
        data_filters.update({column: {'type': col_type,
                                      'min': col_min,
                                      'max': col_max,
                                      'mean': col_mean,
                                      'median': col_median,
                                      'impute_type': col_impute_type,
                                      'imputed_value': col_impute_value}})
    else:
        col_values = og_data[column].unique().tolist()

        data_filters.update({column: {'type': col_type,
                                      'values': col_values}})

with open('./data/metadata.json', 'w') as metadata_file:
    json.dump(data_filters, metadata_file, indent=4)  
    

#select only the wanted columns
required_columns = ['price', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer',
                    'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color', 'lat', 'long']
clean_data = og_data[required_columns]

#ONE HOT ENCODING! WHOOP WHOOP!
categorical_columns = ['manufacturer', 'condition', 'fuel', 'title_status', 'transmission',
                       'drive', 'size', 'type', 'paint_color'] 
enc = OneHotEncoder(sparse_output=False).fit(clean_data[categorical_columns])
encoded = enc.transform(clean_data[categorical_columns])
encoded_df = pd.DataFrame(encoded, columns=enc.get_feature_names_out()) 

non_categorical_columns = ['price', 'year', 'odometer', 'cylinders', 'lat', 'long']

clean_data.reset_index(inplace=True)
clean_data = pd.concat([clean_data[non_categorical_columns].copy(), encoded_df], axis = 1)

column_names = {'features': {col:('numeric' if col in non_categorical_columns else 'onehot')
                             for col in clean_data.columns.to_list()[1:]}, 'label': 'price'}
with open('./data/column_names.json', 'w') as column_names_file:
    json.dump(column_names, column_names_file, indent=4)  

print(' ')
nans = clean_data.isna().sum().to_dict()
n_nans = 0
for col in nans:
    n_nans += nans[col]
    if nans[col] > 0:
        print(f"Detected {nans[col]} NaN values in column '{col}'!")
if n_nans == 0:
    print('Data contains 0 NaN values. :)')


og_data[required_columns].to_parquet(os.path.join(clean_data_folder, cleaned_file_categorical))

clean_data_train, clean_data_test, clean_price_train, clean_price_test = train_test_split(
                    clean_data[clean_data.columns[1:]], clean_data[['price']], 
                    test_size=0.1, random_state=37)
clean_data_train, clean_data_val, clean_price_train, clean_price_val = train_test_split(
                clean_data_train, clean_price_train, 
                test_size=0.1/0.9, random_state=42)

for dataset in [clean_data_train, clean_price_train, clean_data_val, 
                clean_price_val, clean_data_test, clean_price_test]:
    dataset.reset_index(inplace=True, drop=True)

# non_categorical_covs = [ 'year', 'odometer', 'cylinders', 'lat', 'long']
# price_scaler = MinMaxScaler().fit(np.array(clean_price_train).reshape(-1,1))
# clean_price_train = price_scaler.transform(np.array(clean_price_train).reshape(-1,1))
# clean_price_test = price_scaler.transform(np.array(clean_price_test).reshape(-1,1))
# clean_price_val = price_scaler.transform(np.array(clean_price_val).reshape(-1,1))

# scaler = StandardScaler().fit(clean_data_train[non_categorical_covs])
# clean_data_train[non_categorical_covs] = pd.DataFrame(scaler.transform(clean_data_train[non_categorical_covs]), columns= clean_data_train[non_categorical_covs].columns)
# clean_data_train[non_categorical_covs].index = clean_data_train[non_categorical_covs].index
# clean_data_test[non_categorical_covs]  = pd.DataFrame(scaler.transform(clean_data_test[non_categorical_covs]), columns= clean_data_test[non_categorical_covs].columns)
# clean_data_test[non_categorical_covs].index = clean_data_test[non_categorical_covs].index
# clean_data_val[non_categorical_covs]  = pd.DataFrame(scaler.transform(clean_data_val[non_categorical_covs]), columns= clean_data_val[non_categorical_covs].columns)
# clean_data_val[non_categorical_covs].index = clean_data_val[non_categorical_covs].index

clean_data_train.to_parquet(os.path.join(clean_data_folder, cleaned_file_train))
clean_data_test.to_parquet(os.path.join(clean_data_folder, cleaned_file_test))
clean_data_val.to_parquet(os.path.join(clean_data_folder, cleaned_file_val))
clean_price_train.to_parquet(os.path.join(clean_data_folder, cleaned_file_train_price))
clean_price_test.to_parquet(os.path.join(clean_data_folder, cleaned_file_test_price))
clean_price_val.to_parquet(os.path.join(clean_data_folder, cleaned_file_val_price))

print(' ')
print(f'train length: {len(clean_data_train)}, test length: {len(clean_data_test)}, val length: {len(clean_data_val)}.')
