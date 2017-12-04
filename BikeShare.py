import numpy as np 
import pandas as pd
bikes=pd.read_csv('train.csv')
bikes_test=pd.read_csv('test.csv')


bikes.rename(columns={'count':'total'}, inplace=True)
dt=bikes_test['datetime']

from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


temp= pd.DatetimeIndex(bikes['datetime'])
bikes["dayofweek"] = temp.dayofweek
bikes["hour"] = temp.hour
bikes["month"] = temp.month
bikes['year']= temp.year
temp_t=pd.DatetimeIndex(bikes_test['datetime'])
bikes_test["dayofweek"] = temp_t.dayofweek
bikes_test["hour"] = temp_t.hour
bikes_test["month"] = temp_t.month
bikes_test['year']= temp_t.year


import seaborn as sns
sns.pairplot(bikes, x_vars=['season', 'weather','temp', 'humidity'], y_vars='total', kind='reg')


sns.pairplot(bikes, x_vars=['holiday', 'workingday','atemp', 'windspeed'], y_vars='total', kind='reg')


X=bikes.drop(['casual','registered','total','datetime'],axis=1)
y=bikes['total']
X.info()


new_y = np.log(y + 1)


X_train, X_test, y_train, y_test = train_test_split(X, new_y, test_size = 0.33, random_state = 42)


# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
prediction_regr = regr.predict(X_test)
mean_squared_error(y_test, prediction_regr) 


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
prediction = rf.predict(X_test)
mean_squared_error(y_test, prediction)   




rf.fit(X, new_y)
bikes_test= bikes_test.drop(['datetime'],axis=1)
prediction = rf.predict(bikes_test)
prediction = np.exp(prediction) - 1


df=pd.DataFrame({'datetime':dt, 'count':prediction})
df