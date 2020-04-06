"""

Model Optimisation

"""

# Importing

import pandas as pd
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# Specifying Model Parameters

HomeData = pd.read_csv('train.csv')

y = HomeData.SalePrice

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = HomeData[features]

train_x, val_x, train_y, val_y = tts(x,y,random_state = 1)

# Specifying Model

IowaModel = DecisionTreeRegressor(random_state = 1)
IowaModel.fit(train_x,train_y)

ValPredictions = IowaModel.predict(val_x)
Val_mae = mean_absolute_error(val_y, ValPredictions)

print('Validation MAE: {}'.format(Val_mae))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    print(mae)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
l = len(candidate_max_leaf_nodes)
arr = np.zeros(l)

z = 0

for i in candidate_max_leaf_nodes:
    arr[z] = get_mae(i, train_x, val_x, train_y, val_y)
    z += 1
    
MinMAE = min(arr)
print('Minimum MAE is: ' + str(MinMAE))

index = np.argmin(arr)
BestSize = candidate_max_leaf_nodes[index]
print('Best tree size is: ' +str(candidate_max_leaf_nodes[index]))

FinalModel = DecisionTreeRegressor(max_leaf_nodes = BestSize, random_state=0)
FinalModel.fit(train_x,train_y)

