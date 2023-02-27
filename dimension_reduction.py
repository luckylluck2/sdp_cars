import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA


# Load recipes data

data_folder = './data'
recipes_file = 'cleaned_data.parquet'

cars_data = pd.read_parquet(os.path.join(data_folder, recipes_file))

cars_prices = cars_data['price']
cars_data.drop(columns='price', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(cars_data, cars_prices, 
                                                    test_size=0.10, random_state=37)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                    test_size=0.10/0.90, random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed  = scaler.transform(X_test)
X_val_transformed  = scaler.transform(X_val)

pca = PCA(n_components = 2).fit(X_train_transformed)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)
component_threshold_value = [np.flip(np.sort(np.abs(pca.components_[0])))[9], np.flip(np.sort(np.abs(pca.components_[1])))[9]]
interesting_columns = [np.where(np.abs(pca.components_[0]) > component_threshold_value[0]),
                       np.where(np.abs(pca.components_[1]) > component_threshold_value[1])]

print(''.join([f'ColName: {cars_data.columns[i]}; \tWeight: {round(pca.components_[0][i], 2)}\n' \
               for i in interesting_columns[0][0]]))
#beetje useless dit vind ik