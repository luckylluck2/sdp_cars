import numpy as np
import pandas as pd
import os

# Load recipes data

data_folder = './data'
recipes_file = 'recipes.parquet'

og_data = pd.read_parquet(os.path.join(data_folder, recipes_file))
og_data.head(5)

# Select appropriate columns

print('All columns:', og_data.columns.to_list())

required_columns = ['RecipeId', 'CookTime', 'PrepTime', 'TotalTime',
                    'AggregatedRating', 'ReviewCount', 'Calories', 
                    'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent', 
                    'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent', 'RecipeServings']

data = og_data[required_columns]

# Remove rows with missing values

print('Number of rows:', len(data))
print('Number of rows without None values:', len(data.dropna()))

data = data.dropna()

data.head(5)

def timestring_to_minutes(timestring):
    """
    Convert the timestring in the columns 'CookTime', 'PrepTime' or 'TotalTime' to minutes.
    Example: 'PT24H45M' -> 24 * 60 + 45
    
    Args:
        timestring (str): String indicating a time duration
        
    Returns:
        total_time (float): Number of minutes
    """
    if type(timestring) is str:
        # Remove leading 'PT'
        timestring = timestring.replace('PT', '')
        
        # Determine the number of hours from the remaining string before 'H'
        if 'H' in timestring:
            timestring_h = timestring.split('H')
            hours = int(timestring_h[0])
            
            # Consider only the part after 'H' for the minutes
            timestring = timestring_h[1]
        else:
            hours = 0
            
        # Determine the number of minutes from the remaining string before 'M'
        if 'M' in timestring:
            minutes = int(timestring.split('M')[0])
        else:
            minutes = 0
        
        # Calculate the total time duration in minutes
        total_time = hours * 60 + minutes
        return total_time
    else:
        return None

data['CookTime'] = data['CookTime'].map(timestring_to_minutes)
data['PrepTime'] = data['PrepTime'].map(timestring_to_minutes)
data['TotalTime'] = data['TotalTime'].map(timestring_to_minutes)
data = data.astype({'RecipeId': 'int32', 'CookTime': 'int32',
                    'PrepTime': 'int32', 'TotalTime': 'int32'})

print(data.dtypes)
data.head(5)

print(sum((data['Calories'] < 10000) & ((data['PrepTime'] <= 120) & (data['CookTime'] <= 180))))
temp_filter = (data['Calories'] < 10000) & (data['PrepTime'] <= 120) & (data['CookTime'] <= 180) & \
                (data['SodiumContent'] <= 5000)