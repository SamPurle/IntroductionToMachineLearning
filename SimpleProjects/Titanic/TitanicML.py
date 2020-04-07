"""

Titanic:
    
    Predict the survival of passengers on the Titanic using basic Machine Learning concepts/

"""

# Importing Libraries

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Gathering Data

GenderSubmission = pd.read_csv('gender_submission.csv')
Test = pd.read_csv('test.csv')
Train = pd.read_csv('train.csv')

# Converting Columns to Numeric

Train['Sex'] = Train['Sex'].replace(['male','female'],[0,1])
Train['Embarked'] = Train['Embarked'].replace(['Q', 'S', 'C'],[0,1,2])

# Cleaning

Train.dropna(axis = 'rows')

MeanAge = Train.Age.mean()
Train['Age'] = Train['Age'].fillna(MeanAge)

# Specifying Model Parameters

SurvivalFeatures = ['Pclass','Sex','Age','Parch','Fare','Embarked']
MeanAge = Train['Age'].mean()
NullAge = pd.isna(Train['Age'])
x = Train[SurvivalFeatures]

y = Train.Survived

# Building Model

SurvivalModel = RandomForestRegressor(random_state = 1)
SurvivalModel.fit(x,y)

# Converting Columns to Numeric

Test['Sex'] = Test['Sex'].replace(['male','female'],[0,1])
Test['Embarked'] = Test['Embarked'].replace(['Q', 'S', 'C'],[0,1,2])

xtest = Test[SurvivalFeatures]

PredictedY = SurvivalModel.predict(xtest)
PredictedY = np.around(PredictedY,0)
print('The Predicted values are: ')
print(PredictedY)