# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:07:38 2020

@author: justinoberle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Find a slope and intercept for linear trendline to be removed
def best_fit_slope_and_intercept(xs,ys):
    m = (((np.mean(xs)*np.mean(ys)) - np.mean(xs*ys)) /
         ((np.mean(xs)*np.mean(xs)) - np.mean(xs*xs)))
    
    b = np.mean(ys) - m*np.mean(xs)
    
    return m, b


df = pd.read_csv("C:/Python_Scripts/Stock_Predictions/processed_data/TM_Toyota_Motor_Common_Stock_processed_df.csv")
x_array = np.arange(1, df.Close.size + 1)
y_array = np.array(df.Close)

slope, intercept = best_fit_slope_and_intercept(x_array, y_array)

# Remove the trendline
y_subtract = slope * x_array
y_subtracted = y_array - y_subtract

plt.plot(x_array, y_subtracted)

# Add trendline back into data
y_new = y_subtracted + y_subtract

plt.plot(x_array, y_new)