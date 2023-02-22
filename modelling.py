from tabular_data import load_airbnb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor 
from sklearn.preprocessing import StandardScaler


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






