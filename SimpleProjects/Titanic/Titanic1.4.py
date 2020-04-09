"""


Titanic 1.4: Score = 0.7416
    
    String operations to find title of passengers
    
"""

# Import

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Read Data

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

# Feature Engineering

FamSizeTrain = dfTrain['SibSp'] + dfTrain['Parch']
dfTrain['FamSize'] = FamSizeTrain

FamSizeTest = dfTest['SibSp'] + dfTest['Parch']
dfTest['FamSize'] = FamSizeTest


# Specify model parameters

xFeatures = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','FamSize']

xTrain = dfTrain[xFeatures]
xTest = dfTest[xFeatures]

yTrain = dfTrain['Survived']

# Impute columns with Null values

ImpMean = SimpleImputer(strategy = 'mean',copy = False)
ImpMode = SimpleImputer(strategy = 'most_frequent', copy = False)

xTrain['Age'] = ImpMean.fit_transform(xTrain[['Age']])
xTrain['Embarked'] = ImpMode.fit_transform(xTrain[['Embarked']])

xTest['Age'] = ImpMean.fit_transform(xTest[['Age']])
xTest['Fare'] = ImpMean.fit_transform(xTest[['Fare']])

# Convert Names to Titles

NameTrain = xTrain['Name']
NameTest = xTest['Name']

CommaTrain = NameTrain.str.partition(',')
PartTrain = CommaTrain[2]
StopTrain = PartTrain.str.partition('.')
TitleTrain = StopTrain[0]

xTrain['Name'] = TitleTrain

CommaTest = NameTest.str.partition(',')
PartTest = CommaTest[2]
StopTest = PartTest.str.partition('.')
TitleTest = StopTest[0]

xTest['Name'] = TitleTest

# Encode categoric columns

EmbarkedEncTrain = pd.get_dummies(xTrain['Embarked'])
xTrain = xTrain.join(EmbarkedEncTrain)
xTrain = xTrain.drop(columns = 'Embarked')

NameEncTrain = pd.get_dummies(xTrain['Name'])
xTrain = xTrain.join(NameEncTrain)
xTrain = xTrain.drop(columns = 'Name')

SexEncTrain = pd.get_dummies(xTrain['Sex'])
xTrain = xTrain.join(SexEncTrain)
xTrain = xTrain.drop(columns = 'Sex')

EmbarkedEncTest = pd.get_dummies(xTest['Embarked'])
xTest = xTest.join(EmbarkedEncTest)
xTest = xTest.drop(columns = 'Embarked')

NameEncTest = pd.get_dummies(xTest['Name'])
xTest = xTest.join(NameEncTest)
xTest = xTest.drop(columns = 'Name')

SexEncTest = pd.get_dummies(xTest['Sex'])
xTest = xTest.join(SexEncTest)
xTest = xTest.drop(columns = 'Sex')

# Merge columns

TrainCols = xTrain.columns
TestCols = xTest.columns

InTrainNotTest = list(set(TrainCols) - set(TestCols))
InTestNotTrain = list(set(TestCols) - set(TrainCols))

for i in InTestNotTrain:
    xTrain[i] = 0
    
for j in InTrainNotTest:
    xTest[j] = 0
    
# Create and fit model

SurvModel = RandomForestClassifier(random_state = 0)
SurvModel.fit(xTrain,yTrain)

# Predict Survival for test data

yTest = SurvModel.predict(xTest)

# Export to csv

SubCsv = pd.DataFrame({'Survived' : yTest})
SubCsv = SubCsv.set_index(dfTest['PassengerId'],'PassengerId')

SubCsv.to_csv('1.4submission.csv')

