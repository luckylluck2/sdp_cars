import pandas as pd
from sklearn.model_selection import train_test_split

from torch import FloatTensor
from torch.utils.data import Dataset


class CarData():
    def __init__(self):     
        self.label = None
        self.normalize_features = None
        self.label_norm = None
        
        self.train_features = None
        self.numeric_features = None
        self.onehot_features = None
        
        self.categorical_features = None
        
        self.val_size = None
        self.test_size = None
        self.train_size = None
        
        self.feature_means = None
        self.feature_stds = None
        self.label_mean= None
        self.label_std = None
        self.label_max = None
        self.label_min = None
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_from_df(self, data, label, normalize_features=None, val_size=0.1, test_size=0.1, label_norm='standard'):
        self.label = label
        self.normalize_features = normalize_features
        self.label_norm = label_norm
        
        self.val_size = val_size
        self.test_size = test_size
        self.train_size = 1.0 - self.test_size - self.val_size
        
        rel_val_size = self.val_size / (1.0 - self.test_size)
        
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=37)
        train_data, val_data = train_test_split(train_data, test_size=rel_val_size, random_state=42)
        
        self.determine_feature_types(train_data)
        self.set_normalization_params(train_data)
        self.create_datasets(train_data, val_data, test_data)
    
    def load_from_files(self, train_features_file, 
                        train_label_file, 
                        val_features_file, 
                        val_label_file, 
                        test_features_file, 
                        test_label_file,
                        normalize_features,
                        label_norm='standard'):
        train_features_data = pd.read_parquet(train_features_file)
        train_label_data = pd.read_parquet(train_label_file)
        val_features_data = pd.read_parquet(val_features_file)
        val_label_data = pd.read_parquet(val_label_file)
        test_features_data = pd.read_parquet(test_features_file)
        test_label_data = pd.read_parquet(test_label_file)
        
        train_data = pd.concat([train_label_data, train_features_data], axis=1)
        val_data = pd.concat([val_features_data, val_label_data], axis=1)
        test_data = pd.concat([test_features_data, test_label_data], axis=1)
        
        self.label = train_data.columns.to_list()[0]
        self.normalize_features = normalize_features
        self.label_norm = label_norm
        
        self.determine_feature_types(train_data)
        self.set_normalization_params(train_data)
        self.create_datasets(train_data, val_data, test_data)
        
    def determine_feature_types(self, data):
        self.train_features = [col for col in data.columns.to_list() if col != self.label]
        if self.normalize_features is not None:
            self.numeric_features = [col for col in self.train_features if col in self.normalize_features]
        else:
            self.numeric_features = []
        self.onehot_features = [col for col in self.train_features if col not in self.numeric_features]
        self.categorical_features = list(set(['_'.join(col.split('_')[:-1]) for col in self.onehot_features]))
    
    def set_normalization_params(self, train_data):
        if self.normalize_features is not None:
            self.feature_means = train_data[self.normalize_features].mean()
            self.feature_stds = train_data[self.normalize_features].std()
            
        self.label_mean = train_data[self.label].mean()
        self.label_std =  train_data[self.label].std()
        self.label_min = train_data[self.label].min()
        self.label_max = train_data[self.label].max()
    
    def create_datasets(self, train_data, val_data, test_data):                    
        train_data = self.car_ml_data_pipeline(train_data)
        val_data = self.car_ml_data_pipeline(val_data)
        test_data = self.car_ml_data_pipeline(test_data)
        
        self.train_data = CarDataset(train_data)
        self.val_data = CarDataset(val_data)
        self.test_data = CarDataset(test_data)
        
    def car_ml_data_pipeline(self, data):
        if self.normalize_features is not None:
            data[self.normalize_features] = (data[self.normalize_features] - self.feature_means) / self.feature_stds
        
        if self.label_norm == 'standard':
            data[self.label] = (data[self.label] - self.label_mean) / self.label_std
        elif self.label_norm == 'min_max':
            data[self.label] = (data[self.label] - self.label_min) / (self.label_max - self.label_min)
        else:
            raise ValueError(f'Unknown label normalization: {self.label_norm}')
        
        # Select features and label
        X = data.drop(columns=[self.label]).copy().to_numpy()
        y = data[[self.label]].copy().to_numpy()
            
        dataset = [(X[i], y[i]) for i in range(len(X))]
        return dataset

class CarDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        model_input, label = self.dataset[item]
        return FloatTensor(model_input), FloatTensor(label)
