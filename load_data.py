import numpy as np
import pandas as pd
import os

data_folder = './data'
recipes_file = 'recipes.parquet'

data = pd.read_parquet(os.path.join(data_folder, recipes_file))
print(data.head())

print('\n'.join(data['RecipeInstructions'][0]))