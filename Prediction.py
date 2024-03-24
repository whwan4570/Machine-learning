import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Read the data
df = pd.read_csv('./reData/House.csv')

# Convert the 'date' column to a datetime object
df['date'] = pd.to_datetime(df['date'])

fig, axs = plt.subplots(figsize=(16,8), ncols=3, nrows=2)

lm_features = ['InflationRate','InterestRate','MortgageRate','SupplyRate','UnemploymentRate','PopulationGrowth']

for i, feature in enumerate(lm_features):
    row = int(i/3)
    col = i%3

    sns.regplot(x=feature, y='AveragePrice', data=df, ax=axs[row][col])

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# feature, target separate
y_target = df['AveragePrice']
X_data = df[['InflationRate', 'InterestRate', 'MortgageRate', 'SupplyRate', 'UnemploymentRate', 'PopulationGrowth']]


X_train , X_test , y_train , y_test = train_test_split(X_data , y_target , test_size=0.2, random_state=156)

# Linear Regression
lr = LinearRegression()

lr.fit(X_train, y_train)

LinearRegression()

print(X_train.shape, X_test.shape)
print("-------------------------------------------------")
y_preds = lr.predict(X_test)
print(y_preds[0:5])

mse = mean_squared_error(y_test, y_preds)
rmse = np.sqrt(mse)

print(f'MSE : {mse:.3f}, RMSE: {rmse:.3f}')
print(f'Variance score : {r2_score(y_test, y_preds):.3f}')

print("-------------------------------------------------")
print("intercept value with y:", lr.intercept_)

print("coefficient:", np.round(lr.coef_,1))

coeff = pd.Series(data=np.round(lr.coef_, 1), index=X_data.columns)
print(coeff.sort_values(ascending=False))

print("-------------------------------------------------")

from sklearn.model_selection import cross_val_score

# feature, target separate
y_target = df['AveragePrice']
X_data = df[['InflationRate', 'InterestRate', 'MortgageRate', 'SupplyRate', 'UnemploymentRate', 'PopulationGrowth']]

lr = LinearRegression()
lr

LinearRegression()

neg_mse_scores = cross_val_score(lr, X_data, y_target, scoring="neg_mean_squared_error", cv = 5)
rmse_scores =  np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 folds each Negative MSE scores: ', np.round(neg_mse_scores, 2))
print(' 5 folds each RMSE scores : ', np.round(rmse_scores, 2))
print(f' 5 folds average RMSE : {avg_rmse:.3f}')
print("done")