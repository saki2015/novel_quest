# -*- coding: utf-8 -*-
"""

@author: padma
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import (load_iris, load_wine)
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import (GaussianNB, MultinomialNB)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from collections import OrderedDict

from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        classification_report
)

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')
knn = KNeighborsClassifier()
svc_lin = SVC(kernel = 'linear', random_state = 0)
svc_rbf = SVC(kernel = 'rbf', random_state = 0)
nbc = GaussianNB()
mnbc = MultinomialNB()
dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

model_dict = {
    'log_reg': lr,
    'knearest':knn,
    'svc_lin': svc_lin,
    'svc_rbf': svc_rbf,
    'nbc': nbc,
    'dtc':dtc,
    'rfc':rfc

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
            

def tune_parameters(model_name, X, y, nfolds):
    param_grid = {}
    estimator = model_dict[model_name]
    #grid_search = None
    if model_name == 'nbc':
        param_grid = {}
    if model_name == 'svc_rbf':
        Cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': Cs, 'gamma' : gammas}

    if model_name == 'log_reg':
        Cs = [0.001, 0.01, 0.1, 1, 10]
        param_grid = {'C': Cs}
        
    if model_name == 'knearest':
        neighbors = list(range(1, 30))
        ps = [1,2,3,4,5]
        param_grid = {'p': ps, 'n_neighbors': neighbors}
        
    if model_name == 'dtc':
        max_depth = [3,4,5,6,None]
        min_samples_leaf = [1,4,6,8]
        param_grid = {
                'max_depth':max_depth, 
                'min_samples_leaf': min_samples_leaf
        }
    if model_name == 'rfc':
        max_depth = [3,4,5,6,7,8,None]
        min_samples_leaf = [1,4,6,8]
        n_estimators = [10, 20, 30, 50]
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
    
def compare_models(model_dict, X, y, cv_folds=10, metric='accuracy'):
    dict_scores = {}
    for model_name, estimator in model_dict.items():
        scores = cross_val_score(estimator, X, y, cv=cv_folds, scoring=metric)
        avg_score = scores.mean()
        dict_scores[model_name] = avg_score
        
    sorted_scores = sorted(dict_scores.items(), key=lambda x: x[1], reverse=True)
    ord_models_dict = OrderedDict(sorted_scores)
    custom_print(ord_models_dict)
    return sorted_scores
     
def plot_cm(cm):
    n = cm.shape[0]
    df_cm = pd.DataFrame(cm, range(n),range(n))
    plt.figure(figsize = (8,5))
    sns.set(font_scale=1.5)#for label size
    sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, cmap="YlGnBu",
            xticklabels=False, yticklabels=False)# font size
    plt.title('Confusion Matrix')

def load_and_split_data(dataset_name):
    dataset = None
    
    if dataset_name == 'iris':
        dataset = load_iris()
    elif dataset_name == 'wine':
        dataset = load_wine()

    X = dataset.data
    y= dataset.target
    target_names = list(dataset.target_names)
    print('target_names: {0}'.format(target_names))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test, dataset


def tune_model_and_predict(model_name, X_train, y_train, X_test, y_test):
    print('model_name: {0}'.format(model_name))
    best_estimator = tune_parameters(model_name, X_train, y_train, 10) 
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy:{0}'.format(accuracy))
    print('Confusion Matrix:\n{0}'.format(cm))
    print('Classification Report:\n{0}'.format(report))
    
    plot_cm(cm)
    return

#X_train, X_test, y_train, y_test, dataset = load_and_split_data('wine')
X_train, X_test, y_train, y_test, dataset = load_and_split_data('iris')

#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

sorted_models = compare_models(model_dict, X_train, y_train)
for model_name,score in sorted_models[0:2]:
    tune_model_and_predict(model_name, X_train, y_train, X_test, y_test)

