from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import itertools
import pathlib
import joblib
import json
import typing


X, y = load_airbnb('Price_Night')
X.drop(columns='Unnamed: 19', inplace = True)
X['beds'].fillna(1, inplace = True)
X['bathrooms'].fillna(1, inplace = True)
#print(X.isna().sum())
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


# r2_test = r2_score(y_test, y_pred_test)
# # mse_test = mean_squared_error(y_test, y_pred_test)       


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
    #print(hyperparams_names)
    # print(hyperparams_values[0])
    # print(type(hyperparams_values[0]))
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

def tune_regression_model_hyperparameters(model, param_grid, cv = 5):


    grid_search = GridSearchCV(model, param_grid, cv = 5)
    grid_search.fit(X,y)
    # print(f'Gridsearch best parameters {grid_search.best_params_}')
    # print(f'Gridsearch best score {grid_search.best_score_}')
    return (grid_search.best_params_, grid_search.best_score_)



def save_model(folder):
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

# best_params, best_score = custom_tune_regression_model_hyperparameters(SGDRegressor(), param_grid, X, y)
#     #print(best_params, best_score)

def evaluate_all_models():
        
        dt_reg_params = {
        'criterion': ['absolute_error', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'random_state': [42]
        }

   
        rf_reg_params =  {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        gb_reg_params = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.05, 0.1, 0.5],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        # 'max_features': ['auto', 'sqrt']
        }
        
        dt_best_param, dt_best_score = tune_regression_model_hyperparameters(DecisionTreeRegressor(), dt_reg_params)
        #print(f' DT best params {dt_best_param, dt_best_score}')
        
        rf_best_param, rf_best_score = tune_regression_model_hyperparameters(RandomForestRegressor(), rf_reg_params)
        # # print(rf_best_param, rf_best_score)
        #print(f' RF best params {rf_best_param, rf_best_score}')

        gb_best_param, gb_best_score = tune_regression_model_hyperparameters(GradientBoostingRegressor(), gb_reg_params)
        # print(f'GB: {gb_best_param, gb_best_score}')



def find_best_model():
        models = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]

        # evaluate models using cross-validation and RMSE as evaluation metric 
        rmse_scores = []
        for model in models:
            rmse_scores.append(np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)))

        # calculate mean RMSE for each model 
        mean_rmse_scores = []
        for rmse in rmse_scores:
            mean_rmse_scores.append(np.mean(rmse))

        # find the index of the best model based on mean RMSE 
        best_model_index = np.argmin(mean_rmse_scores)

        # print the best model and its mean RMSE 
        print("Best model:", models[best_model_index])
        print("Mean RMSE:", mean_rmse_scores[best_model_index])
        # save_model('./models/regression/linear_regression')


if __name__== "__main__":

    evaluate_all_models()
    find_best_model()





#save_model('models/regression')



        






