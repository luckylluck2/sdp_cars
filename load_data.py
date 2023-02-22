import numpy as np
import pandas as pd
import os

data_folder = './data'
vehicles_file = 'vehicles.csv'
vehicles_parquet = 'vehicles.parquet'

data = pd.read_csv(os.path.join(data_folder, vehicles_file))
print(data.head())

data.to_parquet(os.path.join(data_folder, vehicles_parquet))