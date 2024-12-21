import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset=pd.read_csv(r"C:\Users\MANISHA\Downloads\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

x_train=x_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train) 

y_pred=regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

m_slope=regressor.coef_
print(m_slope)

c_intercept=regressor.intercept_
print(c_intercept)

y_15=m_slope * 15 +c_intercept
y_15

comparison=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparison)

dataset.mean()
dataset['Salary'].mean()

dataset.median()
dataset['Salary'].median()

dataset.mode()
dataset['Salary'].mode()

dataset.std()
dataset['Salary'].std()

dataset.var()
dataset['Salary'].var()

from scipy.stats import variation
variation(dataset.values)
variation(dataset.Salary)

dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

dataset.skew()
dataset['Salary'].skew()

import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

y_mean=np.mean(y)
SSR=np.sum((y_pred-y_mean)**2)
print(SSR)

y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print(SSE)

mean_total=np.mean(dataset.values)
SST=np.sum((dataset.values-mean_total)**2)
print(SST)

r_square=1-(SSR/SST)
r_square

import pickle
filename='linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor,file)
    print('Model has been pickled and saved as linear_regression_model.pkl')
    