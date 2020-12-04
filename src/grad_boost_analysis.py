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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import plot_partial_dependence    

plt.rcParams.update({'font.size': 28})

if __name__ == '__main__':

    gradBoost = pickle.load(open('../model/grad_boost.pkl', 'rb'))
    final_set = pd.read_pickle('../model/rf_data.pkl')

    print('\n DATA COLUMN: ', list(final_set.columns))
    feature_importances = np.argsort(gradBoost.feature_importances_)
    print('FULL FEATURE IMPORTANCES: ', feature_importances)
    print("Top Five:", list(final_set.columns[feature_importances[-1:-6:-1]]))

    number_features = 5
    importances = np.argsort(gradBoost.feature_importances_)
    importance_rank = gradBoost.feature_importances_
    print('\n SECOND TIME FOR IMPORTANCES: ', importances)
    std = np.std( [tree.feature_importances_ for tree in gradBoost.estimators_[:, 0]], axis=0 )
    print('STD: ', std)
    indices = importances[::-1]
    print('INDICES: ', indices)
    features = list(final_set.columns[indices])
    print('FEATURES: ', features)

    print("\n  Feature Ranking:")
    for i in range(number_features):
        print("%d. %s (%f)" % (i + 1, features[i], importance_rank[indices[i]]))

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(range(number_features), 
           importance_rank[indices][:number_features], 
           yerr=std[indices][:number_features], 
           color="cornflowerblue", 
           align="center")
    ax.set_xticks(range(number_features))
    ax.set_xticklabels(features[:number_features], rotation = 45, fontsize=18)
    ax.set_xlim([-1, number_features])
    ax.set_ylabel("Importance", fontsize=22)
    ax.set_xlabel("Features", fontsize=22)
    ax.set_title("Feature Importances - Gradient Boost", fontsize=24)
    fig.tight_layout()
    plt.savefig('../images/feature_importance_gb.png')

    fig, ax = plt.subplots(figsize=(28, 15))
    plot_partial_dependence(gradBoost, final_set, 
                            features[:number_features], 
                            line_kw={"c": "cornflowerblue"},
                            ax=ax)
    ax.set_title("Partial Dependence - Gradient Boost", fontsize=46)
    fig.tight_layout()
    plt.savefig('../images/partial_dependance_gb.png')

