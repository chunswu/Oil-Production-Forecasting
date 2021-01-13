from ploty import *
import numpy as np
import pandas as pd
import pickle
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence
from scipy.stats import randint as sp_randInt 

plt.rcParams.update({'font.size': 18})

def optimizer():
    base_model = RandomForestRegressor(n_estimators=250,
                                       random_state=1)

    base_model.fit(X_train, y_train)

    param_grid = {
    'bootstrap': [True, False],
    'max_depth': sp_randInt(2, 60),
    'max_features': sp_randInt(0, 10),
    'min_samples_leaf': [1, 10, 100],
    'min_samples_split': [4, 5, 6],
    'n_estimators': sp_randInt(10, 600)
    }

    rf = RandomForestRegressor()

    grid_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                            cv = None, n_jobs = -1, n_iter = 10)

    grid_search.fit(X_train, y_train)

    print("\n========================================================")
    print(" Results from Random Search " )
    print("========================================================") 

    print("\n The best estimator across ALL searched params:\n",
          grid_search.best_estimator_)
    best_grid = grid_search.best_estimator_

    print("\n The best score across ALL searched params:\n",
          grid_search.best_score_)

    print("\n The best parameters acros ALL searched params:\n", 
          grid_search.best_params_)

    print("\n ========================================================")

def find_estimators():
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
        randomForest = RandomForestRegressor(n_estimators=tree_num,
                                            n_jobs=-1,
                                            random_state=1)

        randomForest.fit(X_train, y_train)

        model_train_score = randomForest.score(X_train, y_train)

        model_test_score = randomForest.score(X_test, y_test)
        print(tree_num, ' : ', model_test_score)
        model_test_lst.append(model_test_score)

        y_predict = randomForest.predict(X_test)
        model_predict_score = mean_squared_error(y_test, y_predict)
        model_predict_lst.append(model_predict_score**0.5)

    fig, ax = plt.subplots(figsize = (12, 6))
    line_plot(ax, plants, model_test_lst, 'mediumseagreen', 'model test score')
    ax.set_title('Random Forest Model Accuracy', fontsize=34)
    ax.set_xlabel('Number of Estimators', fontsize=24)
    ax.set_ylabel('Root Mean Square Error (%)', fontsize=24)
    ax.legend(loc='lower right', fontsize=20)
    plt.savefig('../images/model_test_rf.png')

    fig, ax = plt.subplots(figsize = (12, 6))
    line_plot(ax, plants, model_predict_lst, 'forestgreen', 'model predict score')
    ax.set_title('Random Forest Predict Error', fontsize=34)
    ax.set_xlabel('Number of Estimators', fontsize=24)
    ax.set_ylabel('Root Mean Square Error (In Barrels)', fontsize=24)
    ax.legend(loc='upper right', fontsize=20)
    plt.savefig('../images/model_predict_rf.png')

if __name__ == '__main__':

    final_set = pd.read_pickle('../model/rf_data.pkl')

    y = final_set.pop('day180').values
    X = final_set.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 1)
    
    randomForest = RandomForestRegressor(n_estimators=350,
                                         n_jobs=-1,
                                         random_state=1)

    randomForest.fit(X_train, y_train)

    model_train_score = randomForest.score(X_train, y_train)
    print('MODEL TRAIN SCORE: ', model_train_score)
    model_test_score = randomForest.score(X_test, y_test)
    print('MODEL TEST SCORE: ', model_test_score)
    y_predict = randomForest.predict(X_test)
    model_predict_score = mean_squared_error(y_test, y_predict)
    print('MODEL PREDICT TEST: ', model_predict_score**0.5)
    '''
    MODEL TRAIN SCORE:  0.968911172673222
    MODEL TEST SCORE:  0.7644400592644415
    MODEL PREDICT TEST:  12220.795660227037
    '''
    find_estimators()

    with open('../model/random_forest.pkl', 'wb') as rf_file:
        pickle.dump(randomForest, rf_file)
    
    for i in range(10):
        optimizer()
    '''
    The best score across ALL searched params:
    0.7644600418098111

    The best parameters acros ALL searched params:
    {'bootstrap': False, 
    'max_depth': 41, 
    'max_features': 3, 
    'n_estimators': 309}
    '''
