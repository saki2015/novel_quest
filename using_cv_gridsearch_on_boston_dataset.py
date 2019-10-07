# -*- coding: utf-8 -*-
"""
Using cross_val_score & gridsearch to pick the best linear regression model
to make predictions with the boston data set

@author: padma
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import (load_boston)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import OrderedDict

from sklearn.metrics import (
       mean_squared_error
)


lr = LinearRegression()
knn = KNeighborsRegressor()
svr = SVR(gamma='scale')

dtr = DecisionTreeRegressor(criterion = 'mse', random_state = 0)
rfr = RandomForestRegressor(n_estimators = 10, criterion = 'mse', random_state = 0)

model_dict = {
    'lin_reg': lr,
    'knearest':knn,
    'svr': svr,
    'dtc':dtr,
    'rfc':rfr,

}
def custom_print(seq, text=None):
    if text:
        print('{0}:'.format(text))
    if isinstance(seq, dict):
        for key, val in seq.items():
            print('{0} --> {1}'.format(key, val))
    elif isinstance(seq, list):
        for el in seq:
            print(el, sep=', ')
            
    
#Use Grid Search to find the best model & the best params
def tune_parameters(model_name, X, y, nfolds):
    param_grid = {}
    estimator = model_dict[model_name]
 
    if model_name == 'svr':
        kernels = ['rbf']
        Cs = [1, 10]
        gammas = [ 0.1, 1]
        epsilons = [0.1]
        param_grid = {'C': Cs, 'gamma' : gammas, 'kernel': kernels, 'epsilon': epsilons}
    
    if model_name == 'lin_reg':
        param_grid = {}
        
    if model_name == 'knearest':
        neighbors = list(range(10, 15, 20, 23, 25))
        ps = [1,2,3,4,5]
        param_grid = {'p': ps, 'n_neighbors': neighbors}
        
    if model_name == 'dtr':
        max_depth = [3,5,6,None]
        min_samples_leaf = [1,6,8]
        param_grid = {
                'max_depth':max_depth, 
                'min_samples_leaf': min_samples_leaf
        }
        
    if model_name == 'rfr':
        max_depth = [3,5,6,8,None]
        min_samples_leaf = [1,6,8]
        n_estimators = [10, 30]
        param_grid = {
                'max_depth':max_depth, 
                'min_samples_leaf': min_samples_leaf,
                'n_estimators':n_estimators,
        }
        
    grid_search = GridSearchCV(estimator, param_grid, cv=nfolds)
    grid_search.fit(X, y)
    print('best_params:{0}'.format(grid_search.best_params_))
    print('best estimator: {0}'.format(grid_search.best_estimator_))
    return grid_search

#Use cross-val_score to order the models according to their negative mse
def compare_models(model_dict, X, y, cv_folds=10, metric='neg_mean_squared_error'):
    dict_scores = {}
    b = y.ravel()
    for model_name, estimator in model_dict.items():
        scores = cross_val_score(estimator, X, b, cv=cv_folds, scoring=metric)
        avg_score = scores.mean()
        dict_scores[model_name] = avg_score
        
    sorted_scores = sorted(dict_scores.items(), key=lambda x: x[1], reverse=True)
    ord_models_dict = OrderedDict(sorted_scores)
    print('Model --> neg_mean_squared_error')
    custom_print(ord_models_dict)
    return sorted_scores
     

def load_and_split_data(dataset_name):
    dataset = None
    
    if dataset_name == 'boston':
        dataset = load_boston()


    X = dataset.data
    y = dataset.target
    feature_names = list(dataset.feature_names)
    #print('feature_names: {0}'.format(feature_names))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test, dataset


def tune_model_and_predict(model_name, X_train, y_train, X_test, y_test):
    print('model_name: {0}'.format(model_name))
    y_train_r = y_train.ravel()
    gscv = tune_parameters(model_name, X_train, y_train_r, 10) 
    best_estimator = gscv.best_estimator_
    print('best estimator:{0}'.format(best_estimator))
    y_pred = best_estimator.predict(X_test)
    
    rmse = np.sqrt(abs(mean_squared_error(y_test, y_pred)))
    
    print('Root Mean Squared Error:\n{0}'.format(rmse))

    return

X_train, X_test, y_train, y_test, dataset = load_and_split_data('boston')

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = sc.fit_transform(y_train.reshape(-1, 1))
y_test = sc.transform(y_test.reshape(-1,1))

#Get the models sorted by neg mse scores
sorted_models = compare_models(model_dict, X_train, y_train)
model_name = sorted_models[0][0] # best regressor with the lowest mse
print('Model with lowest mse: {}'.format(model_name))
tune_model_and_predict(model_name, X_train, y_train, X_test, y_test)
