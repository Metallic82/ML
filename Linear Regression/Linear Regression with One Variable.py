#Linear Regression with One Variable

#Data Preprocessing 

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os 
os.chdir("C:\\Users\\kritanu\\Desktop\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 4 - Simple Linear Regression")
os.getcwd()

#importing the dataset
dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:, 1].values


#splitting dataset into training and test

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Feature Scaling (Not needed here)
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting simple linear regression in the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set results
Y_pred = regressor.predict(X_test)

#Visulaising the Training Set results
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10 #To re-size matplotlib plot
plt.scatter(X_train, Y_train, marker = 'x', color = 'red') #creates scatter plot of training data
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #fits line to scatter
plt.title("Salary vs Experience (Training Set)", fontsize = 'large') #plot title
plt.xlabel("Years of Experience", fontsize = 'large') #x-axis label
plt.ylabel("Salary", fontsize = 'large') #y-label
plt.show() #display plot

#Predicting the Test set results

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.scatter(X_test, Y_test, marker = 'x', color = 'red') #creates scatter plot of test data
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #fits trained line to test data
plt.title("Salary vs Experience (Test Set)") 
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()