import torch
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# from tensorflow.keras.callbacks import TensorBoard
import datetime
import yaml
import warnings
warnings.filterwarnings("ignore")
import time 
import json
import os


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
    def __init__(self, input_dim, hidden_dim, output_dim, depth, **model_configs):
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



def get_nn_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

config_file = 'nn_config.yaml'
config = get_nn_config(config_file)

# Update hyperparameters from config
optimiser = config['optimiser']
learning_rate = config['learning_rate']
hidden_layer_width = config['hidden_layer_width']
depth = config['depth']

# Define keyword arguments for model config initialisation
model_configs = {'config': config}

model = NeuralNetwork(input_dim, hidden_layer_width, output_dim, depth, model_configs=model_configs)
shape = [parameter.shape for parameter in model.parameters()]
#print(shape)

def train(model, data_loader, num_epochs, optimiser, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    loss_values = []
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
            # Normalize the tensor
            #output_normalized = (output - output.min()) / (output.max() - output.min())
            # Normalize the tensor
            #target_normalized = (target - target.min()) / (target.max() - target.min())
            target = target.float()
            print(f'target: {target.size()}')
            loss = loss_fn(output, target.unsqueeze(1))
            loss_values.append(loss.item())
            #print(f'loss_values: {loss_values}')
            loss.backward()
            optimizer.step()

train(model,train_dataloader, 100, optimiser, learning_rate)


def evaluate_model_rmse(model, data_loader):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            total_loss += torch.nn.functional.mse_loss(y_pred, y, reduction='sum').item()
            num_samples += x.size(0)
    avg_loss = total_loss / num_samples
    return avg_loss

def evaluate_model_rmse(model, data_loader):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            total_loss += torch.nn.functional.mse_loss(y_pred, y, reduction='sum').item()
            num_samples += x.size(0)
    avg_loss = total_loss / num_samples
    return avg_loss

def evaluate_model_r2(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in data_loader:
            y_true.append(y.cpu().numpy())
            y_pred.append(model(x).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2

#train(NeuralNetwork, train_dataloader, 100)

def save_model(model, hyperparameters, train_loader, valid_loader, folder):
    """
    Saves a PyTorch model, its hyperparameters, and its performance metrics to disk.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to save.
    - hyperparameters (dict): A dictionary of hyperparameters used to train the model.
    - train_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object representing the training set.
    - valid_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object representing the validation set.
    - test_loader (torch.utils.data.DataLoader): A PyTorch DataLoader object representing the test set.
    """
    # Check if the model is a PyTorch module
    if not isinstance(model, torch.nn.Module):
        raise ValueError("The provided model is not a PyTorch module.")
    
    # Compute training duration
    start_time = time.time()
    model.train()
    for x, y in train_loader:
        y_pred = model(x)
    training_duration = time.time() - start_time
    
    # Compute inference latency
    num_inference_samples = 1000
    inference_start_time = time.time()
    for i, (x, y) in enumerate(valid_loader):
        if i == num_inference_samples:
            break
        y_pred = model(x)
    inference_duration = (time.time() - inference_start_time) / num_inference_samples
    
    # Compute performance metrics
    train_loss = evaluate_model_rmse(model, train_loader)
    valid_loss = evaluate_model_rmse(model, valid_loader)
    
    train_rmse = np.sqrt(train_loss)
    valid_rmse = np.sqrt(valid_loss)
    
    train_r2 = evaluate_model_r2(model, train_loader)
    valid_r2 = evaluate_model_r2(model, valid_loader)
    
    
    # Create directory to save model and metrics
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    directory = os.path.join( 'neural_networks/regression/', timestamp)
    # try:
    #     os.makedirs(directory, exist_ok=True)
    # except OSError as exception:
    #      if exception.errno != errno.EEXIST:
    #             raise
    
    
    # Save PyTorch model
    model_path = f'{folder}/model.pt'
    print(model_path)
    torch.save(model.state_dict(), model_path)
    
    # Save hyperparameters
    hyperparameters_path = f"{folder}/hyperparameters.json"
    with open(hyperparameters_path, "w") as f:
        json.dump(hyperparameters, f)
    
    # Save performance metrics
    metrics = {
        "RMSE_loss": {
            "train": train_rmse,
            "valid": valid_rmse
            
        },
        "R_squared": {
            "train": train_r2,
            "valid": valid_r2
        },
        "training_duration": training_duration,
        "inference_latency": inference_duration
    }
    metrics_path = f"{folder}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

save_model(model, model_configs, train_dataloader, val_dataloader, './neural_networks/regression')






    
