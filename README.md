# Modelling-Airbnb-s-listings
Build a framework to systematically train, tune, and evaluate models on several tasks that are tackled by the Airbnb team. 
This is Airbnb Listing project, in which we will develop a multitude of  classification & regression models and compare their performance across different use cases.  The  Airbnb dataset is loaded which contains both numerical & categorical data and perform cleaning transformations on this data. The proect aim to  address the following:
-Predict the tariff of each listing based on a multitude of features (Regression) 
-Predict distinct Airbnb categories (entire place, apartment, etc.) (Classification)


# Milestone 1 Data Preparation
The data was given in two formats : images and tabular_data. The tabular dataset has the following columns:

  ID: Unique identifier for the listing
  Category: The category of the listing
  Title: The title of the listing
  Description: The description of the listing
  Amenities: The available amenities of the listing
  Location: The location of the listing
  guests: The number of guests that can be accommodated in the listing
  beds: The number of available beds in the listing
  bathrooms: The number of bathrooms in the listing
  Price_Night: The price per night of the listing
  Cleanliness_rate: The cleanliness rating of the listing
  Accuracy_rate: How accurate the description of the listing is, as reported by previous guests
  Location_rate: The rating of the location of the listing
  Check-in_rate: The rating of check-in process given by the host
  Value_rate: The rating of value given by the host
  amenities_count: The number of amenities in the listing
  url: The URL of the listing
  bedrooms: The number of bedrooms in the listing
  
The follow steps were taken for data preparation:
  -Remove missing values
  -The Description column had numerous blank spaces, quotes and string lists that needed to be combined into a single string to provide a comprehensive explanation of    the listing. For this purpose, nested for loops were utilized
 - Created new function called load_airbnb which returns the features and labels of  data in a tuple like (features, labels). The features inlude only numerical data as we going to create Regressor model in next task. 
 
 Features : beds, bathrooms  Cleanliness_rating, Accuracy_rating,  Communication_rating, Location_rating, Check-in_rating,  Value_rating  amenities_count 
 Label/ Target : Price Night

# Milestone 2: Create a Regression model
-Developed a basic regression model with the help of the built-in SGDRegressor model from sklearn to estimate the nightly price of each Airbnb listing. To accomplish this, a Python script called "modelling.py" that imports the "load_airbnb" function and takes the "Price Night" variable as the target label for prediction. 

-Evaluate the regression model performance using RMSE and R2 for training set.  

-Implemented a custom function to perform a grid search over range of hyperparameter values. This function takes in these arguments : model class, training and test sets, dictionary of hyperparamaters. Tuned the range of hyperparameters values of the model using SKLearns gridsearch CV. 

-Created a function save_model to save the regression model, its hyperparameters and the metrics once trained and tuned.

-Evaluate the performance of different models: decision trees, random forests, and gradient boosting under the function named: evaluate_all_models. 

-Created a  function  find_best_model to evaluates which model is best, then returns the loaded model, a dictionary of its hyperparameters, and a dictionary of its performance metrics.

-Out of all model, RandomForestRegressor() was the best model with mean rmse  = 105 


