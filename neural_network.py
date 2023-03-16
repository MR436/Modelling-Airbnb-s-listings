
import torch
#from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import itertools
import datetime
import time
import errno
import json
import os
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
print( type(val_dataloader))
#print(train_dataloader)

# Check it's working
# for batch, (X, y) in enumerate(train_dataloader):
#     print(f"Batch: {batch+1}")
#     print(f"X shape: {X.shape}")
#     print(f"y shape: {y.shape}")
#     break


    

input_dim = 11
#hidden_dim = 2
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
    
#################################  TASK 5 ########################

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

def train(model, data_loader, num_epochs, optimizer, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()
    writer = SummaryWriter()
    loss_values = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0

        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(data)
          
            target = target.float()
    
            loss = loss_fn(output, target.unsqueeze(1))
            loss_values.append(loss.item())
            #print(f'loss_values: {loss_values}')
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            train_loss /= len(train_dataloader.dataset)
            train_acc = 100.0 * train_correct / train_total

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
    #writer.add_graph('Accuracy/Train', train_loss, epoch)

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

def save_model(model, hyperparameters, train_loader, valid_loader):
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
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    folder = 'neural_networks/regression'
    #directory = os.path.join(path, timestamp)
    
    
    # Save PyTorch model
    model_path = f'{folder}/{timestamp}_model.pt'
    print(model_path)
    torch.save(model.state_dict(), model_path)
    
    # Save hyperparameters
    hyperparameters_path = f"{folder}/{timestamp}_hyperparameters.json"
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
    metrics_path = f"{folder}/{timestamp}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

save_model(model, model_configs, train_dataloader, val_dataloader)


############################## Task 7 #########################


def generate_nn_configs(hidden_layers=[64, 128], activations=['relu', 'sigmoid'], 
                        learning_rates=[0.001, 0.01, 0.1], epochs=[10, 50, 100]):
    """
    Generates multiple config dictionaries for a neural network, given a list of possible values for
    the hyperparameters of interest.
    
    Args:
        hidden_layers (list): A list of integers representing the number of neurons in each hidden layer.
        activations (list): A list of strings representing the activation functions to be used in each layer.
        learning_rates (list): A list of floats representing the learning rates to be used for training the network.
        epochs (list): A list of integers representing the number of epochs to be used for training the network.
        
    Returns:
        A list of dictionaries, where each dictionary represents a unique combination of hyperparameters.
    """
    
    # Generate all possible combinations of hyperparameters
    hyperparam_combinations = list(itertools.product(hidden_layers, activations, learning_rates, epochs))
    
    # Create a list of dictionaries, where each dictionary represents a unique combination of hyperparameters
    configs = []
    for combo in hyperparam_combinations:
        config = {
            'hidden_layers': combo[0],
            'activation': combo[1],
            'learning_rate': combo[2],
            'epochs': combo[3]
        }
        configs.append(config)
        
    return configs



def find_best_nn(X_train, y_train, X_val, y_val):
    # Generate all possible configurations
    configs = generate_nn_configs()
    
    # Keep track of the best model's performance
    best_score = 0
    best_model = None
    best_config = None
    
    # Train and evaluate each model configuration
    for config in configs:
        
        # Create the model
        model = nn.Sequential(
            nn.Linear(X_train.shape[1], config['hidden_layers']),
            nn.ReLU(),
            nn.Linear(config['hidden_layers'], 1),
            nn.Sigmoid()
        )

        # Define the loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Train the model
        for epoch in range(config['epochs']):
            optimizer.zero_grad()
            X_train = torch.tensor(X_train).float()
            y_train = torch.Tensor(y_train)
            outputs = model(X_train)
            criterion = nn.BCELoss()
            outputs = torch.Tensor(outputs)
            

            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()

        # Evaluate the model
        with torch.no_grad():
            X_val = torch.tensor(X_val).float()
            y_val = torch.Tensor(y_val)
            predicted = (model(X_val) > 0.5).float()
            score = (predicted == y_val).sum().item() / y_val.size()[0]
        
        # Save the best model and its configuration
        if score > best_score:
            best_score = score
            best_model = model
            best_config = config
            
    # Save the best configuration to a file
    with open('./neural_networks/best_params/hyperparameters.json', 'w') as f:
        json.dump(best_config, f)
    
    return best_model, best_config

find_best_nn(X_train, y_train, X_val, y_val)
