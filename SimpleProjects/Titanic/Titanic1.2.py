"""

Titanic 1.2: Score = 0.73684
    
    Use Simple Imputer to deal with Null values, and One-hot encoding to handle Embarkation
    Potential to include Name, Ticket, and Cabin

"""

# Import

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error


# Read Data

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

# Specify Model Features

xFeatures = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']
xTrain = dfTrain[xFeatures]

yTrain = dfTrain.Survived

xTest = dfTest[xFeatures]

# Impute columns with Null values

NullCols = xTrain.columns[xTrain.isna().any()].tolist()

ImpMean=SimpleImputer(strategy="mean", copy = False )
xTrain['Age'] = ImpMean.fit_transform(xTrain[["Age"]])

ImpMode = SimpleImputer(strategy = 'most_frequent', copy = False)
xTrain['Embarked'] = ImpMode.fit_transform(xTrain[['Embarked']])


# Encoding categorical columns

CatCols = set(xTrain.columns) - set(xTrain._get_numeric_data().columns)

xTrain['Sex'].replace({'male' : 0, 'female' : 1}, inplace = True)

enc = OneHotEncoder(sparse = False)
xTrain['Embarked'] = enc.fit_transform(xTrain[['Embarked']])

# Create and fit model

SurvModel = RandomForestClassifier(random_state = 0)
SurvModel.fit(xTrain,yTrain)
yPred = SurvModel.predict(xTrain)

mae = mean_absolute_error(yTrain,yPred)
print('The Mean Absolute Error for this approach was: ')
print(mae)

# Impute columns with Null values for test data

NullColsT = xTest.columns[xTest.isna().any()].tolist()

ImpMean=SimpleImputer(strategy="mean", copy = False )
xTest['Age'] = ImpMean.fit_transform(xTest[["Age"]])
xTest['Fare'] = ImpMean.fit_transform(xTest[["Fare"]])

# Encoding categorical columns for test data

CatColsT = set(xTest.columns) - set(xTest._get_numeric_data().columns)

xTest['Sex'].replace({'male' : 0, 'female' : 1}, inplace = True)

enc = OneHotEncoder(sparse = False)
xTest['Embarked'] = enc.fit_transform(xTest[['Embarked']])

# Predict Survival for test data

yTest = SurvModel.predict(xTest)

# Export to csv

SubCsv = pd.DataFrame({'Survived' : yTest})
SubCsv = SubCsv.set_index(dfTest['PassengerId'],'PassengerId')
print(SubCsv)

SubCsv.to_csv('submission.csv')