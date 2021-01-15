from ploty import *
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt 

import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.size': 18})

def find_estimators_gradboost():
    '''used to determine the n_estimator value for random forest model
       output plots of the results

    Parameters
    ----------
    NONE
    
    Returns
    -------
    NONE
    '''
    plants = np.arange(50, 2050, 50)
    model_test_lst = []
    model_predict_lst = []

    for tree_num in plants:
        gradBoost = GradientBoostingRegressor(learning_rate=0.1,
                                           n_estimators=tree_num,
                                           random_state=1)

        gradBoost.fit(X_train, y_train)

        model_train_score = gradBoost.score(X_train, y_train)
        model_test_score = gradBoost.score(X_test, y_test)
        print(tree_num, ' : ', model_test_score)
        model_test_lst.append(model_test_score)

        y_predict = gradBoost.predict(X_test)
        model_predict_score = mean_squared_error(y_test, y_predict)
        model_predict_lst.append(model_predict_score**0.5)

    fig, ax = plt.subplots(figsize = (12, 6))
    line_plot(ax, plants, model_test_lst, 'royalblue', 'model test score')
    ax.set_title('Gradient Boost Model Accuracy', fontsize=34)
    ax.set_xlabel('Number of Estimators', fontsize=24)
    ax.set_ylabel('Root Mean Square Error (%)', fontsize=24)
    ax.legend(loc='lower right', fontsize=20)
    plt.savefig('../../images/model_test_gb.png')

    fig, ax = plt.subplots(figsize = (12, 6))
    line_plot(ax, plants, model_predict_lst, 'dodgerblue', 'model predict score')
    ax.set_title('Gradient Boost Predict Error', fontsize=34)
    ax.set_xlabel('Number of Estimators', fontsize=24)
    ax.set_ylabel('Root Mean Square Error (In Barrels)', fontsize=24)
    ax.legend(loc='upper right', fontsize=20)
    plt.savefig('../../images/model_predict_gb.png')

def optimizer():
    base_model = GradientBoostingRegressor(learning_rate=0.1,
                                           n_estimators=500,
                                           random_state=1)

    base_model.fit(X_train, y_train)

    parameters = {'learning_rate'    : sp_randFloat(),
                  'subsample'        : sp_randFloat(),
                  'n_estimators'     : sp_randInt(100, 2000),
                  'max_depth'        : sp_randInt(1, 10),
                  'min_samples_leaf' : sp_randInt(1, 20),
                  'max_features'     : sp_randInt(1, 10) 
                 }

    gradBoost = GradientBoostingRegressor()
    randm = RandomizedSearchCV(estimator=gradBoost, param_distributions = parameters, 
                               cv = None, n_iter = 10, n_jobs=-1)
    
    randm.fit(X_train, y_train)
    
    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================")    
    
    print("\n The best estimator across ALL searched params:\n",
          randm.best_estimator_)
    best_grid = randm.best_estimator_

    print("\n The best score across ALL searched params:\n",
          randm.best_score_)
    
    print("\n The best parameters across ALL searched params:\n",
          randm.best_params_)
    print("\n ========================================================")

if __name__ == '__main__':

    final_set = pd.read_pickle('../../model/data.pkl')
    final_set = final_set.drop('api', axis=1)

    y = final_set.pop('day365').values
    X = final_set.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 1)
    
    gradBoost = GradientBoostingRegressor(learning_rate=0.024,
                                           n_estimators=1156,
                                           max_depth=9,
                                           max_features= 7,
                                           subsample=0.689,
                                           min_samples_leaf=11,
                                           random_state=1)
    gradBoost.fit(X_train, y_train)

    find_estimators_gradboost()

    model_train_score = gradBoost.score(X_train, y_train)
    print('MODEL TRAIN SCORE: ', model_train_score)
    model_test_score = gradBoost.score(X_test, y_test)
    print('MODEL TEST SCORE: ', model_test_score)

    y_predict = gradBoost.predict(X_test)
    model_predict_score = mean_squared_error(y_test, y_predict)
    print('MODEL PREDICT TEST: ', model_predict_score**0.5)

    '''
    MODEL TRAIN SCORE:  0.9316365445409732
    MODEL TEST SCORE:  0.7502904018445549
    MODEL PREDICT TEST:  12582.483865108372
    '''


    with open('../model/grad_boost.pkl', 'wb') as gb_file:
        pickle.dump(gradBoost, gb_file)
    
    for i in range(20):
        optimizer()

    '''
    The best score across ALL searched params:
    0.7787532993136229

    The best parameters across ALL searched params:
    {'learning_rate': 0.024265005054953992, 
    'max_depth': 9, 
    'max_features': 7, 
    'min_samples_leaf': 11, 
    'n_estimators': 1156, 
    'subsample': 0.6890582103303509}
    '''