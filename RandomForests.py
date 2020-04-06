"""

Random Forests

"""

# Importing

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Loading Data

home_data = pd.read_csv('train.csv')

# Specifying Variables

y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Model Specification 

iowa_model = DecisionTreeRegressor(random_state=1)
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

# Fit Data Using Random Forest and Check MAE

from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(random_state = 1)
rfModel.fit(train_X,train_y)

RFval_prediction = rfModel.predict(val_X)
rfModelMAE = mean_absolute_error(RFval_prediction,val_y)

print('Validation MAE for Random Forest Regression: {:,.0f}'.format(rfModelMAE))