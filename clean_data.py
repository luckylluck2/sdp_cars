import numpy as np
import pandas as pd
import os

# Load vehicles data

clean_data_folder = './data'
vehicles_file = 'vehicles.parquet'
cleaned_file = 'cleaned_data.parquet'

og_data = pd.read_parquet(os.path.join(clean_data_folder, vehicles_file))

og_data.columns


# Select appropriate columns

required_columns = ['price', 'year', 'manufacturer', 'condition', 'cylinders', 'fuel', 'odometer',
                    'title_status', 'transmission', 'drive', 'size', 'type', 'paint_color']

clean_data = og_data[required_columns]
# Remove rows with missing values

og_data['cylinders']

def cylinders_to_numeric(cylinderstring):
    """
    Convert # cylinders as a string, to integer value
    """
    if ((cylinderstring == 'other') | (cylinderstring == None)): return None
    else: return(int(cylinderstring.replace(' cylinders', '')))

og_data['cylinders'] = og_data['cylinders'].map(cylinders_to_numeric)
og_data['cylinders'].fillna(og_data['cylinders'].mean(skipna=True))


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

#Apply function to all time-valued columns
clean_data['CookTime'] = clean_data['CookTime'].map(timestring_to_minutes)
clean_data['PrepTime'] = clean_data['PrepTime'].map(timestring_to_minutes)
clean_data['TotalTime'] = clean_data['TotalTime'].map(timestring_to_minutes)
clean_data = clean_data.astype({'RecipeId': 'int32', 'CookTime': 'float64',
                    'PrepTime': 'float64', 'TotalTime': 'float64'})

#Next, we ignore rows with unrealistic values. Any abnormally high values for specific columns are removed.
temp_filter = (clean_data['Calories'] <= 10000) & (clean_data['PrepTime'] <= 120) & (clean_data['CookTime'] <= 180) & \
                (clean_data['SodiumContent'] <= 5000)
clean_data = clean_data[temp_filter]

#remove the TotalTime column, since it is not always sum of prep and cook times. 
clean_data = clean_data.drop(columns='TotalTime')


#Next up, we convert the *Content columns to its respective percentage daily value amount
daily_values = {'Calories': 200,            #gram? (Seems to be calories from fat)
                'FatContent': 65,           #gram
                'SaturatedFatContent': 20,  #gram
                'CholesterolContent': 300,  #milligram
                'SodiumContent': 2400,      #milligram
                'CarbohydrateContent': 300, #gram
                'FiberContent': 25,         #gram
                'SugarContent': 25,         #gram
                'ProteinContent': 50}       #gram 

for columnName in daily_values.keys():
    #Calculate percentage daily value (as decimal value; NOT percentage)
    clean_data[columnName] = clean_data[columnName]/daily_values[columnName]
    #and rename column to include PDV
    newName = columnName.replace('Content', '')
    newName = newName + 'PDV'
    clean_data = clean_data.rename(columns={columnName:newName})


clean_data.to_parquet(os.path.join(clean_data_folder, cleaned_file))
