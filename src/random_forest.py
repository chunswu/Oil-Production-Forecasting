# from functions import *
# from pipeline import *
from ploty import *
# from pyspark.sql.types import *
# from pyspark.sql.functions import struct, col, when, lit
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

plt.rcParams.update({'font.size': 18})

def evaluate(model, test_features, test_labels):
    '''Error calucation used for RandomizedSearchCV

    Parameters
    ----------
    model: DataFrame in pandas
    test_features: DataFrame in pandas
    test_labels: DataFrame in pandas
    
    Returns
    -------
    accuracy: float
    '''
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} barrels.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


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
    line_plot(ax, plants, model_test_lst, 'red', 'model test score')
    ax.set_title('Random Forest Model Error', fontsize=34)
    ax.set_xlabel('Number of Estimators', fontsize=24)
    ax.set_ylabel('Error (%)', fontsize=24)
    ax.legend(loc='lower right', fontsize=20)
    plt.savefig('../images/model_test.png')

    fig, ax = plt.subplots(figsize = (12, 6))
    line_plot(ax, plants, model_predict_lst, 'blue', 'model predict score')
    ax.set_title('Random Forest Predict Error', fontsize=34)
    ax.set_xlabel('Number of Estimators', fontsize=24)
    ax.set_ylabel('Error (In Barrels)', fontsize=24)
    ax.legend(loc='upper right', fontsize=20)
    plt.savefig('../images/model_predict.png')

if __name__ == '__main__':

    final_set = pd.read_pickle('../model/rf_data.pkl')

    # print(final_set.head())
    y = final_set.pop('day180').values
    X = final_set.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 1)
    

    find_estimators()

    randomForest = RandomForestRegressor(n_estimators=350,
                                         n_jobs=-1,
                                         random_state=1)

    randomForest.fit(X_train, y_train)

    

    model_train_score = randomForest.score(X_train, y_train)
    print('MODEL TRAIN SCORE: ', model_train_score)
    model_test_score = randomForest.score(X_test, y_test)
    print('MODEL TEST SCORE: ', model_test_score)
    '''
    limited on unseen data
    '''
    y_predict = randomForest.predict(X_test)
    model_predict_score = mean_squared_error(y_test, y_predict)
    print('MODEL PREDICT TEST: ', model_predict_score**0.5)

    with open('../model/random_forest.pkl', 'wb') as rf_file:
        pickle.dump(randomForest, rf_file)
    
    # # *****************************************
    # # ****    RANDOMIZE GRID SEARCH CV   ******
    # # *****************************************

    base_model = RandomForestRegressor(n_estimators=250,
                                       random_state=1)

    base_model.fit(X_train, y_train)
    base_accuracy = evaluate(base_model, X_test, y_test)

    # Create the parameter grid based on the results of random search 
    param_grid = {
    'bootstrap': [True, False],
    'max_depth': [5, 20, 50, 60],
    'max_features': [None, 2, 5],
    'min_samples_leaf': [1, 10, 100],
    'min_samples_split': [4, 5, 6],
    'n_estimators': [100, 500]
    }
    # Create a based model
    rf = RandomForestRegressor()

    # Instantiate the grid search model
    grid_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
                            cv = 3, n_jobs = -1, verbose = 2)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    grid_search.best_params_
    print(grid_search)

    best_grid = grid_search.best_estimator_
    print(best_grid)
    
    grid_accuracy = evaluate(best_grid, X_test, y_test)

    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    '''
    # RandomForestRegressor(max_depth=60, max_features=None, min_samples_leaf=10,
    #                     min_samples_split=4, n_estimators=500)
    # Model Performance
    # Average Error: 8367.7669 barrels.
    # Accuracy = 75.22%.
    # Improvement of -0.85%.

    '''

