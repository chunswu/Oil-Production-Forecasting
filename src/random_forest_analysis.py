from functions import *
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence    

plt.rcParams.update({'font.size': 18})

if __name__ == '__main__':

    randomForest = pickle.load(open('../model/random_forest.pkl', 'rb'))
    final_set = pd.read_pickle('../model/rf_data.pkl')
    # with open('../model/random_forest.pkl', 'rb') as m:
    #     randomForest = pickle.load(m)

    print('\n DATA COLUMN: ', list(final_set.columns))
    # ['Hybrid', 'Slickwater', 'Gel', 'Latitude', 'Longitude', 'TotalProppant', 'NIOBRARA', 'CODELL', 'COLORADO']
    feature_importances = np.argsort(randomForest.feature_importances_)
    print('FULL FEATURE IMPORTANCES: ', feature_importances)
    # FULL FEATURE IMPORTANCES:  [10  9  8  7  1  3  2  5  4  0  6]
    print("Top Five:", list(final_set.columns[feature_importances[-1:-6:-1]]))
    #  Top Five: ['TotalProppant', 'Latitude', 'Longitude', 'Slickwater', 'Hybrid']

    number_features = 5
    importances = np.argsort(randomForest.feature_importances_)
    importance_rank = randomForest.feature_importances_
    print('\n SECOND TIME FOR IMPORTANCES: ', importances)
    std = np.std([tree.feature_importances_ for tree in randomForest.estimators_],
                axis=0)
    print('STD: ', std)
    indices = importances[::-1]
    print('INDICES: ', indices)
    features = list(final_set.columns[indices])
    print('FEATURES: ', features)

    # Print the feature ranking
    print("\n  Feature Ranking:")
    for i in range(number_features):
        print("%d. %s (%f)" % (i + 1, features[i], importance_rank[indices[i]]))


    # WITH 5
    # Feature Ranking:
    # 1. Latitude (0.179734)
    # 2. Longitude (0.147837)
    # 3. Slickwater (0.123813)
    # 4. Hybrid (0.037648)
    # 5. Gel (0.031699)

    # WITH 6
    # Feature Ranking:
    # 1. TotalProppant (0.458881)
    # 2. Latitude (0.179734)
    # 3. Longitude (0.147837)
    # 4. Slickwater (0.123813)
    # 5. Hybrid (0.037648)


    # print('FIND LOOKS LIKE: ', importance_rank[indices][:number_features])
    # print('FIND LOOKS LIKE: ', std[indices][:number_features])

    # Plot the feature importances of the forest
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(number_features), 
           importance_rank[indices][:number_features], 
           yerr=std[indices][:number_features], 
           color="seagreen", 
           align="center")
    ax.set_xticks(range(number_features))
    ax.set_xticklabels(features[:number_features], rotation = 45, fontsize=18)
    ax.set_xlim([-1, number_features])
    ax.set_ylabel("Importance", fontsize=18)
    ax.set_title("Feature Importances", fontsize=24)
    fig.tight_layout()
    plt.savefig('../images/feature_importance.png')

    fig, ax = plt.subplots(figsize=(28, 10))
    plot_partial_dependence(randomForest, final_set, 
                            features[:number_features], 
                            line_kw={"c": "seagreen"},
                            ax=ax)
    ax.set_title("Partial Dependence", fontsize=34)
    fig.tight_layout()
    plt.savefig('../images/partial_dependance.png')


    
    # *****************************************
    # ****    RANDOMIZE GRID SEARCH CV   ******
    # *****************************************

    # base_model = RandomForestRegressor(n_estimators=250,
    #                                    random_state=1)

    # base_model.fit(X_train, y_train)
    # base_accuracy = evaluate(base_model, X_test, y_test)

    # # Create the parameter grid based on the results of random search 
    # param_grid = {
    # 'bootstrap': [True, False],
    # 'max_depth': [5, 20, 50, 60],
    # 'max_features': [None, 2, 5],
    # 'min_samples_leaf': [1, 10, 100],
    # 'min_samples_split': [4, 5, 6],
    # 'n_estimators': [100, 500]
    # }
    # # Create a based model
    # rf = RandomForestRegressor()

    # # Instantiate the grid search model
    # grid_search = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, 
    #                         cv = 3, n_jobs = -1, verbose = 2)

    # # Fit the grid search to the data
    # grid_search.fit(X_train, y_train)

    # grid_search.best_params_
    # print(grid_search)

    # best_grid = grid_search.best_estimator_
    # print(best_grid)
    
    # grid_accuracy = evaluate(best_grid, X_test, y_test)

    # print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

    '''
    RandomForestRegressor(max_depth=60, max_features=None, min_samples_leaf=10,
                        min_samples_split=4, n_estimators=500)
    Model Performance
    Average Error: 9331.6155 barrels.
    Accuracy = 68.28%.
    Improvement of -5.91%.
    '''

