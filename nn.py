import torch
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from tensorflow.keras.callbacks import TensorBoard
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

input_dim = 11
hidden_dim = 2
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        print(x.shape)
        print(self.layer_1)
        print(self.layer_2)
        x = torch.nn.functional.relu(self.layer_1(x.float()))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x

model = NeuralNetwork(input_dim, hidden_dim, output_dim)
shape = [parameter.shape for parameter in model.parameters()]
print(shape)

def train(model, data_loader, num_epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    loss_values = []
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            target = target.float()
            print(f'target: {target.size()}')
            loss = loss_fn(output, target.unsqueeze(1))
            loss_values.append(loss.item())
            #print(f'loss_values: {loss_values}')
            loss.backward()
            optimizer.step()

train(model, train_dataloader, 100)

# def get_nn_config(config_file):
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#     return config

# config_file = 'nn_config.yaml'
# config = get_nn_config(config_file)

# Pass the config as hyperparameters to your model
#model = NeuralNetwork(config=config)

# # Train your model
# class NNModel(nn.Module):
#     def __init__(self, config):
#         super(NNModel, self).__init__()
#         self.config = config
#         self.hidden_layer_width = config['hidden_layer_width']
#         self.depth = config['depth']
#         #self.optimiser = torch.optim.SGD(self.parameters(), lr=config['learning_rate'])

#         # Define your model architecture using the config
#         # self.layers = nn.ModuleList()
#         input_size = 11
#         in_features = input_size
#         for i in range(self.depth):
#             out_features = self.hidden_layer_width
#             self.layers.append(nn.Linear(in_features, out_features))
#             self.layers.append(nn.ReLU())
#             in_features = out_features
#         self.layers.append(nn.Linear(in_features, 1))
# model = NNModel(config=config)
# train(model,train_dataloader, 100)
# # import torch.optim as optim

# # def train_model(config):
# #     # Get hyperparameters from the configuration dictionary
# #     optimizer_name = config['optimiser']
# #     learning_rate = config['learning_rate']
# #     hidden_layer_width = config['hidden_layer_width']
# #     depth = config['depth']
    
# #     # Instantiate your model with the configuration
# #     model = NNModel(config)
    
# #     # Define optimizer using the specified name and learning rate
# #     optimizer_class = getattr(optim, optimizer_name)
# #     optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    
# #     # Train your model using the specified hyperparameters
# #     # for epoch in range(100):
# #     #     # ...
# #     #     # Use the hidden_layer_width and depth hyperparameters to define your model architecture
# #     #     # ...
# #     #     optimizer.zero_grad()
# #     #     # ...
# #     #     # Perform forward and backward pass using the model
# #     #     # ...
# #     #     optimizer.step()
# #     #     # ...