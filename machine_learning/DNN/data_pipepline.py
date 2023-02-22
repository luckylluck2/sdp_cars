import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from torch import FloatTensor
from torch.utils.data import Dataset

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

def recipes_data_pipeline(cleaned_data_file, train=True):
    data = pd.read_parquet(cleaned_data_file)
    X = data[['CookTime', 'PrepTime', 'CaloriesPDV', 'FatPDV', 
              'SaturatedFatPDV', 'CholesterolPDV', 'SodiumPDV', 
              'CarbohydratePDV', 'FiberPDV', 'SugarPDV', 'ProteinPDV']].to_numpy()
    y = data[['AggregatedRating', 'ReviewCount']].to_numpy()
    
    n_rows = len(X)
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std
    
    prior_rating = 3.0
    prior_weight = 5.0
    y_bayes = ((prior_rating * prior_weight) + (y[:, 0] * y[:, 1])) / (y[:, 1] + prior_weight)
    
    y = np.array([y_bayes <= 2.0, (2.0 < y_bayes) & (y_bayes < 4.0), y_bayes >= 4.0]).astype(int).T
    # condlist = np.array([y_bayes <= 2.0, (2.0 < y_bayes) & (y_bayes < 4.0), y_bayes >= 4.0]).astype(int)
    # encodings = [np.tile(np.array([1, 0, 0]), (n_rows, 1)), 
    #              np.tile(np.array([0, 1, 0]), (n_rows, 1)), 
    #              np.tile(np.array([0, 0, 1]), (n_rows, 1))]
    # encodings = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    # y = np.choose(condlist, encodings)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    if train:
        dataset = [(X_train[i], y_train[i]) for i in range(len(X_train))]
    else:
        dataset = [(X_test[i], y_test[i]) for i in range(len(X_test))]
    return dataset

class FoodDataset(Dataset):
    def __init__(self, data_file, train=True):
        self.dataset = recipes_data_pipeline(data_file, train=train)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        model_input, label = self.dataset[item]
        return FloatTensor(model_input), FloatTensor(label)
