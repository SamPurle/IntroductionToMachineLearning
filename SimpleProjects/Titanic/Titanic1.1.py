"""

Titanic 1.1: Score = 0.74162
    
    Predict the survival of passengers on baord the titanic using intermediate machine learning principles
    
    
"""

# Importing

import pandas as pd 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# Reading Data

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')
Ans = pd.read_csv('gender_submission.csv')

# Specify model parameters

y = dfTrain.Survived

# Cabin Removed - potential to include with string operations 
# Embarked removed - potential to include with null cleaning

xFeatures = ['Pclass','Sex','SibSp','Parch','Fare','Age','Embarked']
x = dfTrain[xFeatures]

# Find columns with null values

NullCols = x.columns[x.isna().any()].tolist()

MeanAge = x.Age.mean()
x.Age = x.Age.fillna(MeanAge)

# Find categorical columns

Cols = x.columns
NumCols = x._get_numeric_data().columns
CatCols = set(Cols) - set(NumCols)

SexCleanup = {'male' : 0, 'female' : 1}
x.Sex.replace(SexCleanup, inplace = True)

EmCleanup = {np.NaN : 0, 'S' : 0, 'C' : 1, 'Q' : 2}
x.Embarked.replace(EmCleanup, inplace = True)

# Creating and fitting model

SurvivalModel = RandomForestRegressor(random_state = 0)
SurvivalModel.fit(x,y)
yPred = SurvivalModel.predict(x)
yInt = yPred > 0.5
SurvPreds = yInt.astype(int).tolist()

mae = mean_absolute_error(y,SurvPreds)

# Generating Predictions for Test Data

xTest = dfTest[xFeatures]

# Find columns with null values

NullCols = xTest.columns[xTest.isna().any()].tolist()

MeanAge = xTest.Age.mean()
xTest.Age = xTest.Age.fillna(MeanAge)

MeanFare = xTest.Fare.mean()
xTest.Fare = xTest.Fare.fillna(MeanFare)

# ModePort = x.Embarked.mode()
# x.Embarked = x.Embarked.fillna(ModePort)


# Find categorical columns

Cols = xTest.columns
NumCols = xTest._get_numeric_data().columns
CatCols = set(Cols) - set(NumCols)

SexCleanup = {'male' : 0, 'female' : 1}
xTest.Sex.replace(SexCleanup, inplace = True)

xTest.Embarked.replace(EmCleanup, inplace = True)

# Creating and fitting model

yPredTest = SurvivalModel.predict(xTest)
yIntTest = yPredTest > 0.5
SurvPredsTest = yIntTest.astype(int).tolist()

Ans['Preds'] = pd.Series(SurvPredsTest)

maeAns = mean_absolute_error(Ans.Survived,Ans.Preds)
print('The average error for this solution is: ')
print(maeAns)

SubCsv = pd.DataFrame({'Survived' : SurvPredsTest})
SubCsv = SubCsv.set_index(Ans.PassengerId,'PassengerId')
print(SubCsv)

SubCsv.to_csv('1.1submission.csv')