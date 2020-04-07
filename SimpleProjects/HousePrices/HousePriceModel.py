"""

House Prices:
    
    Predict the prices of Houses based on machine learning regression models.

"""

# Importing

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Loading Data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
val = pd.read_csv('sample_submission.csv')

xFeatures = ['LotArea','OverallQual','OverallCond','YearBuilt','1stFlrSF','2ndFlrSF']
x = train[xFeatures]

xval = test[xFeatures]
yval = val['SalePrice']

y = train['SalePrice']

PriceModel = RandomForestRegressor(random_state = 1)
PriceModel.fit(x,y)

print('Predicting Sale Prices on the test data: ')
ypred = PriceModel.predict(xval)
print(ypred)

print('The actual Sale Prices were: ')
print(yval)

