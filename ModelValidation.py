"""

Model Validation


"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error as mae

HomeData=pd.read_csv('train.csv')

y = HomeData.SalePrice

HomeDataFeatures = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

x = HomeData[HomeDataFeatures]

train_x, val_x, train_y, val_y = tts(x,y, random_state=0)

IowaModel = DecisionTreeRegressor()
IowaModel.fit(train_x,train_y)

print('First Predictions:')
print(IowaModel.predict(train_x.head()))

print('Actual Prices:')
print(y.head().tolist())

MeanAvgError = mae(val_y,IowaModel.predict(val_x))
print(MeanAvgError)