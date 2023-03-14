import torch
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
#from tensorflow.keras.callbacks import TensorBoard
import datetime
import yaml
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('./clean_tabular_data.csv')
features = data.select_dtypes(include=['int64', 'float64']).drop(columns = 'Price_Night')#.values
features.fillna(1, inplace=True)
features = features.values
labels = data['Price_Night'].values

X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=.33, random_state=26)
#print(y_train.shape)

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.x[idx], dtype=torch.float64)
        label = torch.tensor(self.y[idx], dtype=torch.float64)
        return features, label

# Define batch size
batch_size = 64

train_data = AirbnbNightlyPriceRegressionDataset(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

val_data = AirbnbNightlyPriceRegressionDataset(X_val, y_val)
val_dataloader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True)
#print(train_dataloader)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


    