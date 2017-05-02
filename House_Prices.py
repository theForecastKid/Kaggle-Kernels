import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import skew
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


tr = pd.read_csv('train.csv')
ts = pd.read_csv('test.csv')
tr.dropna()
ts.dropna()

y = tr['SalePrice']
X = tr.drop('SalePrice', axis = 1)

print("Let's see the price histogram")
plt.hist(y, bins = 50)
plt.show()
print('Skewness:', skew(y))

print('Now we will apply a log transform to make it more Gaussian')
y_log = np.log(y)
plt.hist(y_log, bins = 50)
plt.show()
print('Skewness:', skew(y_log),'\n')

# join Xtr and Xts
df = X.append(ts)

#Find features with skewed histogram and apply log trnsform
numeric = df.dtypes[df.dtypes != "object"].index
skewed = tr[numeric].apply(lambda x: skew(x)) #calculate skewness
skewed = skewed[skewed > 0.5]
skewed = skewed.index
df[skewed] = np.log(df[skewed] + 1)
df = df.fillna(df.mean())

#Now lets encode string features
df = pd.get_dummies(df)

#Now let's split data into train and test again
Xtr = df[:tr.shape[0]]
Xts = df[tr.shape[0]:]

print('We can start to use some models now...\n')
print('Model - RidgeRegression')
model = Ridge()
param_grid = {'alpha' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}
grid = GridSearchCV(model, cv = 4, n_jobs = -1, param_grid = param_grid)
print('Fitting model for all possible alpha values...')
grid.fit(Xtr, y_log)
print(pd.DataFrame(grid.cv_results_), '\n')
print('Using best alpha to predict...\n')
yp_log = grid.best_estimator_.predict(Xts)
yp_RR = np.exp(yp_log)
y_stack1 = np.exp(grid.best_estimator_.predict(Xtr))

print ('Model - RandomForest')
model = RandomForestRegressor(500)
param_grid = {'max_depth' : [38, 40, 42], 'max_features': [35, 40, 45]}
grid = GridSearchCV(model, cv = 4, n_jobs = -1, param_grid = param_grid)
print('Fitting model for all possible values...')
grid.fit(Xtr, y_log)
print(pd.DataFrame(grid.cv_results_), '\n')
print('Using best parameters to predict...\n')
yp_log = grid.best_estimator_.predict(Xts)
yp_RF = np.exp(yp_log)
y_stack2 = np.exp(grid.best_estimator_.predict(Xtr))

print('Model - PCA + Linear Regression')
pipe = Pipeline([
    ('pca', PCA()),
    ('regressor', LinearRegression())
])
N_COMPONENTS = [130, 135, 140, 145, 150, 155, 160]
param_grid = {'pca__n_components': N_COMPONENTS}
grid = GridSearchCV(pipe, cv = 4, n_jobs = -1, param_grid = param_grid)
print('Fitting model for all possible values...')
grid.fit(Xtr, y_log)
print(pd.DataFrame(grid.cv_results_), '\n')
print('Using best parameters to predict...\n')
yp_log = grid.best_estimator_.predict(Xts)
yp_LR = np.exp(yp_log)
y_stack3 = np.exp(grid.best_estimator_.predict(Xtr))

Xtr = Xtr.add(y_stack1, axis=0).add(y_stack2, axis=0).add(y_stack3, axis=0)
Xts = Xts.add(yp_RR, axis =0).add(yp_RF, axis=0).add(yp_LR, axis =0)

print('Now we can use satck all models with GradientBoosting\n')
print ('Model - GradientBoosting')
model = GradientBoostingRegressor(n_estimators=1000)
param_grid = {'max_depth' : [3, 4, 5], 'learning_rate': [0.001, 0.002, 0.004, 0.005, 0.006, 0.008, 0.01]}
grid = GridSearchCV(model, cv = 4, n_jobs = -1, param_grid = param_grid)
print('Fitting model for all possible values...')
grid.fit(Xtr, y_log)
print(pd.DataFrame(grid.cv_results_), '\n')
print('Using best parameters to predict...\n')
yp_log = grid.best_estimator_.predict(Xts)
yp_GB = np.exp(yp_log)


pd.DataFrame({'Id': ts.Id, 'SalePrice': yp_GB}).to_csv('ensemble.csv', index =False)
