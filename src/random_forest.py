from functions import *
from pipeline import *
from ploty import *
from pyspark.sql.types import *
from pyspark.sql.functions import struct, col, when, lit
import numpy as np
import pandas as pd
import random
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence


# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#     return accuracy


def find_estimators():

    plants = np.arange(50, 1550, 50)
    model_test_lst = []
    model_predict_lst = []

    for tree_num in plants:
        randomForest = RandomForestRegressor(n_estimators=tree_num,
                                            n_jobs=1,
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

    spark = (ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("sparkSQL exercise") 
        .getOrCreate()
        )
    sc = spark.sparkContext

    df = spark.read.csv('../data/dj_basin.csv',
                         header=True,
                         quote='"',
                         sep=",",
                         inferSchema=True)
    df.createOrReplaceTempView("data")

    fluid_data = spark.sql("""
                    SELECT
                        api,
                        State,
                        TotalCleanVol,
                        FluidVol1,
                        UPPER(FluidType1) AS fluid_type1,
                        FluidVol2,
                        UPPER(FluidType2) AS fluid_type2,
                        FluidVol3,
                        UPPER(FluidType3) AS fluid_type3,
                        FluidVol4,
                        UPPER(FluidType4) AS fluid_type4,
                        FluidVol5,
                        UPPER(FluidType5) AS fluid_type5
                    FROM data
                    """)

    parameter_data = spark.sql("""
                        SELECT 
                            api,
                            Latitude, 
                            Longitude,
                            UPPER(formation) AS formation,
                            TotalProppant,
                            Prod180DayOil AS day180
                        FROM data
                        """)

                            # Prod365DayOil AS day365,
                            # Prod545DayOil AS day545,
                            # Prod730DayOil AS day730,
                            # Prod1095DayOil AS day1095,
                            # Prod1460DayOil AS day1460,
                            # Prod1825DayOil AS day1825,


    fluid_data = clean_data(fluid_data)
    final_set = finished_form(fluid_data, parameter_data)

    formation_seperate = ['NIOBRARA', 'CODELL']
    state_seperate = ['COLORADO', 'WYOMING']

    for layers in formation_seperate:
        final_set = column_expand(final_set, 'formation', layers)

    for state in state_seperate:
        final_set = column_expand(final_set, 'State', state)

    final_set = final_set.drop(columns=['formation'])
    final_set = final_set.drop(columns=['State'])
    final_set = final_set.drop(columns=['TotalCleanVol'])
    final_set = final_set.dropna()
    final_set = final_set.set_index('api')

    # print(final_set.head())
    y = final_set.pop('day180').values
    X = final_set.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 1)
    

    find_estimators()

    randomForest = RandomForestRegressor(n_estimators=350,
                                        n_jobs=1,
                                        random_state=1)

    randomForest.fit(X_train, y_train)

    model_train_score = randomForest.score(X_train, y_train)
    print('MODEL TRAIN SCORE: ', model_train_score)

    model_test_score = randomForest.score(X_test, y_test)
    print('MODEL TEST SCORE: ', model_test_score)

    y_predict = randomForest.predict(X_test)
    model_predict_score = mean_squared_error(y_test, y_predict)
    print('MODEL PREDICT TEST: ', model_predict_score**0.5)    


    print('\n DATA COLUMN: ', list(final_set.columns))
    # ['Hybrid', 'Slickwater', 'Gel', 'Latitude', 'Longitude', 'TotalProppant', 'NIOBRARA', 'CODELL', 'COLORADO', 'WYOMING']
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
    # 6. Gel (0.031699)

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

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_partial_dependence(randomForest, final_set, 
                            features[:number_features], 
                            line_kw={"c": "seagreen"},ax=ax)
    ax.set_title("Partial Dependence", fontsize=24)
    ax.xaxis.label.set_size(50)
    ax.yaxis.label.set_size(50)
    fig.tight_layout()
    plt.savefig('../images/partial_dependance.png')

    # base_model = RandomForestRegressor(n_estimators=1000,
    #                                    random_state=1)
    # base_model.fit(X_train, y_train)
    # base_accuracy = evaluate(base_model, X_test, y_test)
    
    # # Create the parameter grid based on the results of random search 
    # param_grid = {
    # 'bootstrap': [True],
    # 'max_depth': [5, 20, 50, 60],
    # 'max_features': [2, 3],
    # 'min_samples_leaf': [3, 4, 5],
    # 'min_samples_split': [4, 5, 6],
    # 'n_estimators': [100, 500, 1000]
    # }
    # # Create a based model
    # rf = RandomForestRegressor()

    # # Instantiate the grid search model
    # grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
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
    RandomForestRegressor(max_depth=50, max_features=3, min_samples_leaf=3,
                      min_samples_split=6, n_estimators=500)
    Model Performance
    Average Error: 9096.4644 degrees.
    Accuracy = 69.70%.
    Improvement of -5.51%.
    '''

