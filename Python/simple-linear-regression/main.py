"""simple linear regression model"""
"""import the needed libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
"""read the data file using pandas"""
dataset = pd.read_csv("Salary_Data.csv")
"""matrix of features"""
x = dataset.iloc[:, :-1].values
"""dependent variable"""
y = dataset.iloc[:, -1].values
print(x)
print(y)
'''split the dataset into train and test sets'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 1)
print(x_train)
"""training the simple linear regression model on the testing set"""
regressor = LinearRegression()
regressor.fit(x_train, y_train)
"""predicting the test results
returns a vector of predictions"""
y_pred = regressor.predict(x_test)
"""visualize the training set results
puts red points into a 2d plot
x-axis is the number of experience - the feature
y-axis is the dependant variable - salary"""
plt.scatter(x_train, y_train, color = "red")
"""plot the function's curve"""
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary vs Experience. Training set.")
plt.xlabel("years of experience")
plt.ylabel('salary')
plt.show()
"""visualizing the test results"""
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary vs Experience")
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()

