# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 23:37:18 2021

@author: justinoberle
"""

from matplotlib import pyplot as plt
import numpy as np

a = [85,86,82,89,91,80]
b = [84,86,87,89,91,85]
a = actual
b = predicted
x = np.arange(1, len(a) + 1)
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

def NormalizedDifference(actual, predicted):
    #n means normalized
    actual_n = []
    predicted_n = []
    difference = []
    for i in range(len(actual)):
        a_n = (actual[i] - min(actual)) / (max(actual) - min(actual))
        p_n = (predicted[i] - min(actual)) / (max(actual) - min(actual))
        actual_n.append(a_n)
        predicted_n.append(p_n)
        diff = np.abs(a_n - p_n)
        difference.append(diff)
    average_diff = sum(difference) / len(difference)
    return average_diff

def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    b = np.mean(ys) - m*np.mean(xs)
    return m, b

def correct_direction(slope_a, slope_b):
    # 0=none, 1=not bad but not good, 2=good
    if slope_a >= .1:
        if slope_b <= -.1:
            direction = 0
        elif slope_b > -.1 and slope_b < .1:
            direction = 1
        else:
            direction = 2
    elif slope_a > -.1 and slope_a < .1:
        if slope_b <= -.1:
            direction = 1
        elif slope_b > -.1 and slope_b < .1:
            direction = 2
        else:
            direction = 0
    elif slope_a <= -.1:
        if slope_b <= -.1:
            direction = 2
        elif slope_b > -.1 and slope_b < .1:
            direction = 1
        else:
            direction = 0
    return direction

numbers = []
for i in range(len(a)): 
    if b[i] <= .00001:
        pass
    else:
        if b[i] > a[i]:
            number = (b[i] - a[i]) / b[i] 
            numbers.append(number)
        else:
            number = (a[i] - b[i]) / b[i] 
            numbers.append(number)
if len(numbers) > 0:    
    stat = sum(numbers) /len(numbers)
else:
    stat = "invalid"
rmse = getRMSE(b, a)
stat2 = NormalizedDifference(a,b)
slope_a, intercept_a = best_fit_slope_and_intercept(x, a)
slope_b, intercept_b = best_fit_slope_and_intercept(x, b)
is_good = correct_direction(slope_a, slope_b)
if is_good == 0:
    rmse = 0
    stat = 0
    stat2 = 0
plt.plot(a)
plt.plot(b)