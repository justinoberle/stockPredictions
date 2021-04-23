# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:09:55 2021

@author: justinoberle
"""

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("C:/Python_Scripts/Stock_Predictions/processed_data/AAPL_Apple_processed_df.csv")
# create X array which is 2D array but only one array inside. This is x values of known data.
x_array = np.arange(1, df.Close.size + 1)
# create y array which is 1D array. This is y values of known data.
y_array = np.array(df.Close)

def get_begin_index_randomly(quantity, y_array, df, interval):
    starting_index = round(len(y_array) - len(y_array) / 2)
    array = np.arange(starting_index, len(y_array) - interval)
    random_starting_points = np.random.choice(array, size=quantity, replace=True)
    return random_starting_points
    
def get_interval_data(quantity, y_array, df):
    interval = 7
    begin_indices = get_begin_index_randomly(quantity, y_array, df, interval)
    list_of_y_arrays = []
    list_of_y_arrays_validate = []
    list_of_x_arrays = []
    list_of_x_arrays_validate = []
    list_of_ending_indices = []
    for i in range(len(begin_indices)):
        begin_index = begin_indices[i]
        end_index = begin_index + interval
        list_of_ending_indices.append(end_index)
        y_array_new = y_array[:begin_index]
        y_array_validate = y_array[begin_index:end_index]
        x_array_new = x_array[:begin_index]
        x_array_validate = x_array[begin_index:end_index]

        list_of_y_arrays.append(y_array_new)
        list_of_y_arrays_validate.append(y_array_validate)
        list_of_x_arrays.append(x_array_new)
        list_of_x_arrays_validate.append(x_array_validate)
    
    return list_of_y_arrays, list_of_y_arrays_validate, list_of_x_arrays, list_of_x_arrays_validate, list_of_ending_indices
def getRMSE(actual, predicted):
    length = len(actual)
    result_list = []
    for i in range(length):
        difference = predicted[i] - actual[i]
        result = (difference * difference)
        result_list.append(result)
    summation = np.sum(result_list)
    final = np.sqrt(summation / length)
    return final


number_of_tries = 1
y_arrays, y_arrays_validate, x_arrays, x_arrays_validate, ending_indices = get_interval_data(number_of_tries, y_array, df)

y_arrays = y_arrays[0]
y_arrays_validate = y_arrays_validate[0]
x_arrays = np.array(x_arrays).T
x_arrays_validate = np.array(x_arrays_validate).T




















# Create a random dataset
#rng = np.random.RandomState(1)
#X = np.sort(5 * rng.rand(80, 1), axis=0)
#y = np.sin(X).ravel()
#y[::5] += 3 * (0.5 - rng.rand(16))

X = x_arrays
y = y_arrays

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0, len(x_arrays) + 210, 1)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()