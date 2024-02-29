import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

data = pd.read_csv('gld_price_data.csv')

# print(data.head())
# print(data.tail())

# print(data.isnull().sum())
# print(data.describe())

# ======================================================================

#### Correlation: ####

# corr = data.corr()
# plt.figure(figsize = (8,8))
# sns.heatmap(corr, cbar=True, square=True, fmt='.1f',annot=True,annot_kws={'size':8},cmap='Red')
# print(corr['GLD'])

#### Splitting the data: ####


# X = data.drop(['Date','GLD'],axis=1)
# Y = data['GLD']

# X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.15,random_state=2)

# #### training the model: ####

# model = RandomForestRegressor(n_estimators=100)

# model.fit(X_train,Y_train)

# test_data_prediction = model.predict(X_test)
# # print(test_data_prediction)

# ##### Comparing using R squared error: #####

# error_score = metrics.r2_score(Y_test, test_data_prediction)
# print(error_score)

# =================================================================================

########################--------------- Mine ---------------########################

X = data.drop(['Date','GLD','SPX', 'USO', 'EUR/USD'],axis=1)
Y = data['GLD']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.175,random_state=2)

model = RandomForestRegressor(n_estimators=100)

model.fit(X_train,Y_train)

pred = model.predict(X_test)
print(pred)

error_score = metrics.r2_score(Y_test, pred)
print(error_score)