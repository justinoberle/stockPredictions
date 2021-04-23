# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:37:42 2020

@author: justinoberle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


load_data_bool = True
filename = "AMD_3_30_21"

def load_data(filename):
    df = pd.read_csv("C:/Python_Scripts/Stock_Predictions/Raw_Data/" + str(filename) + ".csv")
    return df

if load_data_bool == True:
    df = load_data(filename)
    df = df.sort_index(ascending=False)

df2 = df.copy()
for i in range(len(df.columns)):
    df[df.columns[i]] = df[df.columns[i]].values[::-1]

# df['Close/Last'] = df['Close/Last'].str.replace('$', '')
# df['Close/Last'] = df['Close/Last'].astype(float)
# df['Open'] = df['Open'].str.replace('$', '')
# df['Open'] = df['Open'].astype(float)
# df['High'] = df['High'].str.replace('$', '')
# df['High'] = df['High'].astype(float)
# df['Low'] = df['Low'].str.replace('$', '')
# df['Low'] = df['Low'].astype(float)
# df['Close'] = df['Close'].str.replace('$', '')
# df['Close'] = df['Close'].astype(float)

df = df.rename(columns={"Close/Last": "Close", 'Open': 'Open', 'High': 'High', 'Low': 'Low'})
x = np.arange(0, len(df.Date))
y_close = df['Close']
y_open = df['Open']
y_high = df['High']
y_low = df['Low']

plt.plot(x, y_close, label='Close')
plt.plot(x, y_open, label='Open')
plt.plot(x, y_high, label='High')
plt.plot(x, y_low, label='Low')
plt.legend()

df.to_csv("C:/Python_Scripts/Stock_Predictions/processed_data/" + filename + "_processed_df.csv", index=False)

