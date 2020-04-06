"""

Explore Your Data

"""

# Importing Libraries

import math
import pandas as pd
from datetime import date
from sklearn.tree import DecisionTreeRegressor

# Iowa training set

dfI = pd.read_csv('train.csv')

MeanArea = math.ceil(dfI['LotArea'].mean())

NewestDate = dfI['YearBuilt'].max()
CurYear = date.today().year

NewestAge = CurYear - NewestDate

# Melbourne dataset

dfM = pd.read_csv('melb_data.csv')

y = dfM.Price

MelbFeatures = ['Rooms','Bathroom','Landsize','Longtitude','Lattitude']
x = dfM[MelbFeatures]

MelbModel = DecisionTreeRegressor(random_state = 1)
MelbModel.fit(x,y)

print('Making predictions for the following 5 houses:')
print(x.head())

print('The Price predictions are:')
print(MelbModel.predict(x.head()))