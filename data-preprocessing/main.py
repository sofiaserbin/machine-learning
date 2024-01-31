"""import libraries"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
"""importing the dataset"""
dataset = pd.read_csv("C:/Users/Acer/Python/machine-learning/machine-learning/data-preprocessing/Data.csv")
print(dataset)
"""creating the matrix of features - first 3 columns
the left number in the range is excluded
we take all values in all columns except for the last column 
because the last column is the dependant variable vector"""
x = dataset.iloc[:, :-1].values
print(x)
"""creating the dependant variable data
-1 without : means there is no range used
only the last column will be stored in the variable"""
y = dataset.iloc[:, -1].values
print(y)
"""taking care of missing data
replacing missing data by the average of the values in the column
np.nan means empty"""
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)
"""encoding categorical data
here we can include a binary vector for each country"""
"""encoding the independent variable - country - with one hot encoder
has to be a numpy array for further model training"""
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
print(x)
"""encode the dependant variable - with label encoder
it maps yes to 1 and no to 0
doesn't have to be a numpy array for further model training"""
le = LabelEncoder()
y = le.fit_transform(y)
print(y)
"""splitting the data set into the training and test sets
feature scaling comes afterwards"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print(x_test)
print(x_train)
print(y_test)
print(y_train)
"""feature scaling"""
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.fit_transform(x_test[:, 3:])
print(x_train)
print(x_test)


