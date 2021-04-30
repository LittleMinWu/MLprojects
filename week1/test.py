import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#read data
df = pd.read_csv("week1.csv")
print(df.head())

x = np.array(df.iloc[:, 0])
x = x.reshape(-1, 1)
df1 = pd.read_csv("week2.csv")
print(df.head())

x1 = np.array(df.iloc[:, 0])
x1 = x1.reshape(-1, 1)
print(x[0])
print(x1[0])