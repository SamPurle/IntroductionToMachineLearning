"""

Model Building Exercise

"""

# Importing

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Gathering Data

HomeData = pd.read_csv('train.csv')

# Specifying Model Paramaters

y = HomeData.SalePrice

IowaFeatures = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
x = HomeData[IowaFeatures]

IowaModel = DecisionTreeRegressor(random_state = 1)
IowaModel.fit(x,y) 

print('Making predictions on the following houses:')
print(x.head())

print('The predictions are:')
print(IowaModel.predict(x.head()))