"""


Titanic 1.5: Score = 0.63636
    
    Using OneHotEncoder and df merging
    
"""

# Import

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error as mae

# Read Data

dfTrain = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')

# Concatenate

ColMerge = set(dfTrain.columns) - set(dfTest.columns)

for i in ColMerge:
    dfTest[i] = -1
    
df = pd.concat([dfTrain,dfTest])
df['FamSize'] = df['SibSp'] + df['Parch']

# Convert Names to Titles

Name = df['Name']

Comma = Name.str.partition(',')
Part = Comma[2]
Stop = Part.str.partition('.')
Title = Stop[0]

df['Title'] = Title

# Specify model features

xFeatures = ['PassengerId','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title','FamSize']
x = df[xFeatures]

y = dfTrain['Survived']

# Impute null columns

NullCols = x.columns[x.isna().any()].tolist()

ImpMean = SimpleImputer(copy = False, strategy = 'mean')
ImpMode = SimpleImputer(copy = False, strategy = 'most_frequent')

x['Age'] = ImpMode.fit_transform(x[['Age']])
x['Fare'] = ImpMode.fit_transform(x[['Fare']])
x['Embarked'] = ImpMode.fit_transform(x[['Embarked']])

# Encode categorical columns

CatCols = set(x.columns) - set(x._get_numeric_data().columns)

enc = OneHotEncoder(sparse = False)

EmbEnc = enc.fit_transform(x[['Embarked']])
EmbNames = enc.get_feature_names(['Embarked'])
Emb = pd.DataFrame(EmbEnc)
Emb.columns = EmbNames
x = x.join(Emb)
x = x.drop(columns = 'Embarked')

SexEnc = enc.fit_transform(x[['Sex']])
SexNames = enc.get_feature_names(['Sex'])
Sex = pd.DataFrame(SexEnc)
Sex.Columns = SexNames
x = x.join(Sex)
x = x.drop(columns = 'Sex')

TitEnc = enc.fit_transform(x[['Title']])
TitNames = enc.get_feature_names(['Title'])
Tit = pd.DataFrame(TitEnc)
Tit.columns = TitNames
x = x.join(Tit)
x = x.drop(columns = 'Title')

# Create and fit model

xTrain = x[x['PassengerId'] <= 891]
xTest = x[x['PassengerId'] >= 892]

SurvModel = RandomForestClassifier(random_state = 40, n_estimators = 250, criterion = 'entropy')
SurvModel.fit(xTrain,y)

# Predict Survival for test data

yTest = SurvModel.predict(xTest)
yTrain = SurvModel.predict(xTrain)

print('The predicted survivor count is: ')
print(yTest.sum())

print('MAE for the Train set was:')
print(mae(y,yTrain))
# Export to csv

SubCsv = pd.DataFrame({'Survived' : yTest})
SubCsv = SubCsv.set_index(dfTest['PassengerId'],'PassengerId')

SubCsv.to_csv('1.5submission.csv')
