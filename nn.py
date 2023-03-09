import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class AirbnbNightlyPriceRegressionDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('./clean_tabular_data.csv')
        self.features = features = self.data.select_dtypes(include=['int64', 'float64']).drop(columns = 'Price_Night')#.values
        self.features.fillna(1, inplace=True)
        self.features = self.features.values
        self.labels = self.data['Price_Night'].values
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float64)
        label = torch.tensor(self.labels[idx], dtype=torch.float64)
        return features, label
    
# def __init__(self):
    #     # load the csv file into a pandas dataframe
    #     self.data = pd.read_csv('./clean_tabular_data.csv')

    # def __len__(self):
    #     return len(self.data)

    # def __getitem__(self, index):

        # get the features and label for the sample at the given index
        # features = self.data.select_dtypes(include=['int64', 'float64']).drop(columns = 'Price_Night')]
        # #label = self.data['price_per_night']
        # #label = self.data.iloc[index]['price_per_night']
        # print(features)

        # # convert features and label to torch tensors
        # features = torch.tensor(features.values, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.float32)

        # return features, label
    
# Define the proportions for train, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Get the total number of samples in the dataset
dataset = './clean_tabular_data.csv'
dataset_size = len(dataset)
print(dataset_size)

# Calculate the number of samples for each set
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# Use the random_split function to split the dataset into train, validation, and test sets
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Define batch size
batch_size = 32

# Create a data loader for the train set that shuffles the data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create a data loader for the validation set that doesn't shuffle the data
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create a data loader for the test set that doesn't shuffle the data
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dataset = AirbnbNightlyPriceRegressionDataset()
features, labels = dataset[0]
print(f'features: {features}')




