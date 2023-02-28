from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import itertools
import pathlib
import joblib
import json


X, y = load_airbnb('Price_Night')
X.drop(columns='Unnamed: 19', inplace = True)
X['beds'].fillna(1, inplace = True)
X['bathrooms'].fillna(1, inplace = True)
print(X.isna().sum())
#print(X.bathrooms.unique())


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.3)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
#print(X_train)
#Instantiate & Train the model
sgdR_model = SGDRegressor()
sgdR_model.fit(X_train, y_train)

y_pred = sgdR_model.predict(X_test)

#y_pred_test = sgdR_model.predict(X_test)

r2_test= r2_score(y_test, y_pred)
mse_test = mean_squared_error(y_test, y_pred)
# print(f'This is my r2_score: {r2_test}')
# print(f'This is my mse_score: {mse_test}')

#slope = sgdR_model.coef_
# y_intercept = sgdR_model.intercept_
# print(y_intercept)
# r2_test = r2_score(y_test, y_pred_test)
# # mse_test = mean_squared_error(y_test, y_pred_test)       
# 



def custom_tune_regression_model_hyperparameters(model, hyperparams, X, y, test_size =0.3):
    # Split the data into training and validation sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    # Create a list of all possible combinations of hyperparameters.
    #print(hyperparams)
    hyperparams_values = list(itertools.product(*hyperparams.values()))
    #print(hyperparams_values)
    #hyperparams_values = [*hyperparams.values()]
    
    #hyperparams_values = list(*hyperparams.values())
    hyperparams_names = list(hyperparams.keys())
    
    #print(hyperparams_values[0])
    #print(type(hyperparams_values[0]))
    # Initialize variables to store the best hyperparameters and performance metric.
    best_params = {}
    best_score = 0
    
    # For each combination of hyperparameters:
    for values in hyperparams_values:
        params = dict(zip(hyperparams_names, values))
        print(params)
        
        # Train the model on the training set.
        model.set_params(**params)
        model.fit(X_train, y_train)
        
        # Evaluate the model on the validation set using accuracy.
        y_pred = model.predict(X_test).round()
        score = accuracy_score(y_test, y_pred)
        print(score)
        
        # Store the performance metric for that combination of hyperparameters.
        if score > best_score:
            best_params = params
            best_score = score
    
    return best_params, best_score

param_grid = {
    'alpha': 10.0 ** -np.arange(1, 7),
    'loss': ['squared_error', 'huber', 'squared_epsilon_insensitive', 'epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'learning_rate': ['constant', 'optimal', 'invscaling'],}

def tune_regression_model_hyperparameters():
    param_grid = {
        'alpha': 10.0 ** -np.arange(1, 7),
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['constant', 'optimal', 'invscaling'],
    }
    grid_search = GridSearchCV(sgdR_model, param_grid, cv = 5)
    grid_search.fit(X,y)
    print(f'Gridsearch best parameters {grid_search.best_params_}')
    print(f'Gridsearch best score {grid_search.best_score_}')

def save_model(folder):
    print('hi')
    file = pathlib.Path(f'{folder}/model.joblib')
    print(file)
    joblib.dump(sgdR_model, file )

    hyperparameters = {
        'learning_rate': 0.01,
        'num_epochs': 100,
        'batch_size': 32,
        'hidden_units': [64, 32],
    }

    # save to file using JSON
    with open(f'{folder}/hyperparameters.json', 'w') as f:
        json.dump(hyperparameters, f)

    #Store the metrics in a dictionary
        y_pred = sgdR_model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        metrics = {
        'mse': mse,
        'r2': r2
        }
    #Save the metrics to a file in JSON format
    with open(f'{folder}/metrics.json', 'w') as f:
        json.dump(metrics, f)

    #best_params, best_score = custom_tune_regression_model_hyperparameters(SGDRegressor(), param_grid, X, y)
    #print(best_params, best_score)
#save_model('models/regression')



        






