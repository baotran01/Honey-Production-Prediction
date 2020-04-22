import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Read data and plot a scatter plot with values as total production per year
df = pd.read_csv("honeyproduction.csv")
prod_per_year = df.groupby('year').totalprod.mean().reset_index()
X = prod_per_year["year"]
X = X.values.reshape(-1,1)
y = prod_per_year["totalprod"]
plt.scatter(X,y)
plt.ticklabel_format(useOffset=False, style='plain')

# Use Linear Regression to predict the honey production from 2013 to 2050 and plot out the result
regr = LinearRegression()
regr.fit(X,y)
y_predict = regr.predict(X)
X_future = np.array(range(2013,2051))
X_future = X_future.reshape(-1,1)
future_predict = regr.predict(X_future)
plt.plot(X,y_predict)
plt.plot(X_future,future_predict)
plt.show()
