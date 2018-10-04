# Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

titanic_data = pd.read_csv("titanic.csv")


# Data Wrangling

titanic_data.drop('Cabin', axis=1, inplace=True)
titanic_data.dropna(inplace=True)


sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
embarked = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
pclass = pd.get_dummies(titanic_data['Pclass'], drop_first=True)

titanic_data = pd.concat([titanic_data, sex, embarked, pclass], axis=1)
titanic_data.drop(['PassengerId', 'Pclass', 'Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

# Performing Logistic Regression

X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=2)

regression = LogisticRegression()
regression.fit(X_train, y_train)

predictions = regression.predict(X_test)
print(predictions)

accuracy = regression.score(X_train, y_train)

report = classification_report(y_test, predictions)

matrix = confusion_matrix(y_test, predictions)

prediction_score = accuracy_score(y_test, predictions)


