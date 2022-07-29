# -*- coding: utf-8 -*-
# # Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('house-prices.csv')

print(dataset)

print(dataset.shape)

print(dataset.info())

datatypes = dataset.dtypes
print(datatypes)

# Data visualization
from matplotlib import pyplot
dataset.hist()
pyplot.show()

# Data visualization
dataset.plot(kind='density' ,subplots=True, layout=(3,3), sharex=False)
pyplot.show()

#Checking Missing values
print(dataset.isnull().sum())

print(dataset.mode())

mean = dict(dataset.mean())
data=dataset.fillna(value=mean)

print(mean)

print(data.isnull().sum())

# Extracting dependent and independent variables:
# Extracting independent variable:
X = data.iloc[:, 2:3].values 
# Extracting dependent variable:
y = data.iloc[:, 1].values # this is for 1 d array  coming up this will make it 2-d array

print(X)

print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
print(y_pred)

from sklearn import metrics
print("MAE %2.f" %(metrics.mean_absolute_error(y_test,y_pred)))
print("RMSE %2.f" %(np.sqrt(metrics.mean_absolute_error(y_test,y_pred))))

print('Train Score: %f' %(regressor.score(X_train, y_train))) 
print('Test Score: %f' % (regressor.score(X_test, y_test)) )

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(regressor,open('house_price.pkl', 'wb')) 
# Saving model to disk

