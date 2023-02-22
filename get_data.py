import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Create a directory to store the data in
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Use the Kaggle API to download recipe and review data
api = KaggleApi()

# By default, the authenticate function looks for a '~/.kaggle/kaggle.json' file
# to connect to your Kaggle account. 
api.authenticate()
api.dataset_download_files('austinreese/craigslist-carstrucks-data', 
                           path=data_dir, 
                           quiet=False, # show download progress
                           unzip=True) # unzip the downloaded archive

data_folder = './data'
vehicles_file = 'vehicles.csv'
vehicles_parquet = 'vehicles.parquet'

data = pd.read_csv(os.path.join(data_folder, vehicles_file))

data.to_parquet(os.path.join(data_folder, vehicles_parquet))