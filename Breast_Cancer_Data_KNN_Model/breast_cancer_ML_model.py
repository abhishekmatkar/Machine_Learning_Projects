# Essential imports required for our machine learning model

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Get Breast Cancer Dataset from this link>>>
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

# 1) Read Breast Cancer Dataset using Pandas library, remember keep this dataset in the
#    folder where your program resides.
breast_cancer_data = pd.read_csv('breast-cancer-wisconsin.data')

# See the data on print screen, you can comment it later by adding "#" before statement

print("The first 5 rows of the dataset:- \n", breast_cancer_data.head())

# Shape of Data by rows & columns, you can comment it later by adding "#" before statement

print("Shape of our Dataset in (rows, columns):- \n", breast_cancer_data.shape)

# Now if you look into Data description of the link provided above their it's written that
# the missing values contains '?' so to handle this we have to fill it with ignored number

breast_cancer_data.replace('?', -99999, inplace=True)

# So in the above line we have handled the missing values across dataset, now if
# we look at data we have to separate it into features & labels but here
# column "id" cannot be feature as it has no contribution in predicting
# whether cancer is benign or malign, so we will simply drop that column

breast_cancer_data.drop(['id'], 1, inplace=True)

# Now after dropping id we have 9 features columns and last column is our label having two values 2 & 4
# where 2 for benign and 4 for malign, our final goal is predict which type of cancer it is 2 or 4

# Now before applying any machine learning algorithm we have to convert this data into an array
# Here we use Numpy library to convert this dataset into an array

# For features we will name this variable as X

X = np.array(breast_cancer_data.drop(['class'], 1))  # Here we have dropped "class" column as it is our label

# for label we will name variable as y

y = np.array(breast_cancer_data['class'])

# our dataset is now converted to arrays, Now we will split our dataset into training and testing,
# Here we will use 80% of data as training and 20% for testing to specify this we will give
# attribute test_size=0.2
# for this we will import train_test_split from sklearn, look into imports

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Finally our training and testing dataset is ready, now we will use K Nearest Neighbours (KNN)
# algorithm to train this dataset. So for this we have imported neighbours from
# sklearn library

knn_algo = neighbors.KNeighborsClassifier()

# Now we will fit this model in variable we just created and we will fit training data

knn_algo.fit(X_train, y_train)

# Our dataset is finally trained , now let's check the confidence in percentage of this training model
# we just created

confidence = knn_algo.score(X_test, y_test)
print("Confidence of Model:- \n", confidence)

# yay!! our dataset is almost 97% accurate

# Now we can either predict whether breast cancer is malign or benign i.e 4 & 2 by
# using "predict" method of KNN, or we can manually feed the data, here we will manually feed the data

example_features = np.array(([4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 1, 3, 2, 2, 1, 4, 1, 2]))

# In this example_features we gave two data feed to predict in array format

prediction = knn_algo.predict(example_features)


print("Prediction:- \n", prediction)

# Here we successfully predicted that this cancer is a benign cancer.










