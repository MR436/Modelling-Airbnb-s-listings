from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from modelling import save_model
import numpy as np


X, y = load_airbnb('Category')
X.drop(columns='Unnamed: 19', inplace = True)
X['beds'].fillna(1, inplace = True)
X['bathrooms'].fillna(1, inplace = True)
print(X, y)
#print(X.bathrooms.unique())


# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)  
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
# Create logistic regression object 
log_reg_model = LogisticRegression()  
# Fit the model with training data 
log_reg_model.fit(X_train, y_train)  
# # Predict on testing data 
y_pred = log_reg_model.predict(X_test) 

# calculate accuracy, precision, recall, and F1 score 
accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred, average='micro') 
recall = recall_score(y_test, y_pred, average='micro') 
f1 = f1_score(y_test, y_pred, average='micro')  
print(f"Accuracy: {accuracy}" ) 
print(f"Precision: {precision}" ) 
print(f"Recall: {recall}") 
print(f"F1 score: {f1}")

def tune_classification_model(model, hyperparams, X, y, cv=5):    
    # Create a GridSearchCV object     
    grid_search = GridSearchCV(model, hyperparams)   
    # Fit the GridSearchCV object to the data     
    grid_search.fit(X, y)          
    # Return the best estimator 
    print(f'Best estimator{grid_search.best_estimator_}, Best Score {grid_search.best_score_}')
    return (grid_search.best_estimator_ , grid_search.best_score_)

save_model('./models/classification')

def evaluate_all_models():    
    dt_class_params = {'max_depth': [2, 4, 6, 8, 10],
                        'min_samples_split': [2, 4, 6, 8, 10],
                        'min_samples_leaf': [1, 2, 3, 4, 5]     
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
        'min_samples_leaf': [1, 2, 4],     
        'max_features': ['auto', 'sqrt']     
        }      

def find_best_model():
    dt_best_param, dt_best_score = tune_classification_model(DecisionTreeClassifier(), dt_class_params, X, y, cv=5)     
    print(dt_best_param, dt_best_score)     
    rg_best_param, rg_best_score = tune_classification_model(RandomForestClassifier(), rf_reg_params, X, y, cv=5)     
    print(rg_best_param, rg_best_score)      
    gb_best_param, gb_best_score = tune_classification_model(GradientBoostingClassifier(), gb_reg_params, X, y, cv=5)     
    print(gb_best_param, gb_best_score)   

if __name__== "__main__":

    evaluate_all_models()
    find_best_model()


