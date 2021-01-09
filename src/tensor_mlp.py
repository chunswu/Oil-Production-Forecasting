# mlp for regression
from numpy import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
# import sys
# sys.setrecursionlimit(15000)

if __name__ == '__main__':
    # load the dataset
    path = '../model/data.csv'
    df = pd.read_csv(path, sep='\t', index_col=False)
    # df.set_index(keys='api', inplace=True)
    df.drop('api', axis=1, inplace=True)
    print(df.head(5))
    # split into input and output columns
    X = df.values[:, :-1]
    X_f = df.columns
    all_features = X_f[1:-1]
    print(all_features)
    y = df.values[:, -1]
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)
    print(X_train)
    print('shape of splits: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # determine the number of input features
    n_features = X_train.shape[1]
    print('N FEATURES: ', n_features)
    # define model
    model = Sequential()
    # kernel_init = keras.initializers.variance_scaling(seed=88)
    # RMSE: 28036.468
    kernel_init = keras.initializers.GlorotUniform(seed=88)
    # RMSE: 26606.376
    # kernel_init = keras.initializers.GlorotNormal(seed=88)
    # RMSE: 26917.145
    # kernel_init = keras.initializers.glorot_normal(seed=88)
    # RMSE: 27513.185
    # kernel_init = keras.initializers.glorot_uniform(seed=88)
    # RMSE: 29466.417
    # kernel_init = keras.initializers.RandomNormal(seed=88)
    # RMSE: 28750.373
    # kernel_init = keras.initializers.RandomUniform(seed=88)
    # RMSE: 28538.575
    # kernel_init = keras.initializers.truncated_normal(seed=88)
    # RMSE: 28422.978
    # kernel_init = keras.initializers.Identity
    # RMSE: 28654.134
    # kernel_init = keras.initializers.lecun_normal(seed=88)
    # RMSE: 80563.449
    # kernel_init = keras.initializers.lecun_uniform(seed=88)
    # RMSE: 81805.432
    # kernel_init = keras.initializers.random_normal(seed=88)
    # RMSE: 28005.506
    # kernel_init = keras.initializers.random_uniform(seed=88)
    # RMSE: 52264.639
    # kernel_init = keras.initializers.variance_scaling(seed=88)
    # RMSE: 79370.741
    # kernel_init = keras.initializers.Zeros
    # RMSE: 80271.013
    # kernel_init = keras.initializers.zeros
    # RMSE: 81938.751
    # kernel_init = keras.initializers.Ones
    # N/A
    # kernel_init = keras.initializers.Orthogonal(seed=88)
    # RMSE: 79512.972
    # kernel_init = keras.initializers.constant
    # RMSE: 80292.017
    model.add(Dense(8, activation='relu', kernel_initializer=kernel_init, input_shape=(n_features,)))
    # model.add(Dense(1000, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.05))
    model.add(Dense(500, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.05))
    model.add(Dense(100, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.15))
    model.add(Dense(75, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.15))
    model.add(Dense(50, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.15))
    model.add(Dense(25, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dropout(0.05))
    # model.add(Dense(60, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dense(30, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dense(15, activation='relu', kernel_initializer=kernel_init))
    # [3, 20, 10, 5] - RMSE: 26606.376
    # [3, 30, 15, 5] - RMSE: 26945.793
    # [3, 40, 30, 10] - RMSE: 26853.574
    # [5, 40, 30, 10] - RMSE: 26823.950
    # [10, 60, 30, 15] - RMSE: 29198.843
    # [5, 60, 30, 15, 1, 5, 60, 30, 15] - RMSE: BAD
    # [5, 60, 60, 60] - RMSE: 25943.007
    # [5, 70, 70, 70] - RMSE: 28441.942
    # [5, 65, 65, 65] - RMSE: 26522.423
    # [5, 62, 62, 62] - RMSE: 26993.181
    # [5, 58, 58, 58] - RMSE: 26138.409
    # [1, 60, 60, 60] - RMSE: 30070.733
    # [6, 60, 60, 60] - RMSE: 26484.99
    # [8, 100, 60, 60] - RMSE: 26186.08
    # [8, 200, 75, 50, 25] - RMSE: 27679.00
    # [8, 300, 150, 75, 50, 25] - RMSE: 27202.03
    # [8, 300, 100, 75, 50, 25] - RMSE: 28157.26
    # [8, 500, 100, 75, 50, 25] - RMSE: 25476.13
    # [8, 1000, 500, 100, 75, 50, 25] - RMSE: 27241.77
    # [8, 500, 100, 75, 50, 25]*DROP - RMSE: 
    model.add(Dense(1))
    # compile the model
    # optimizer='adam'
    # optimizer='rmsprop'
    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer='adam', loss='mse', metrics='mse')
    # fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=2, validation_split=0.4, verbose=1)
    # evaluate the model
    error = model.evaluate(X_test, y_test, steps=1, use_multiprocessing=True, verbose=1)
    print('THIS IS THE ERROR: ', error)
    print('MSE: %.2f, RMSE: %.2f' % (error[0], sqrt(error[0])))

    # y_true = predictions.select('label').toPandas()
    # y_pred = predictions.select('prediction').toPandas()
    # r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    # print('r2_score: {:4.3f}'.format(r2_score))

    # permutation importance
    # partial dependance plots

    # mlp_model = KerasRegressor(build_fn=model)
    # mlp_model.fit(X_test, y_test)
    # perm = PermutationImportance(model, scoring='accuracy', random_state=88).fit(X_test, y_test, epochs=2, batch_size=2)
    # perm = PermutationImportance(model, scoring="balanced_accuracy")
    # perm.fit(X_test, y_test)   
    
    # eli5.show_weights(perm, feature_names=all_features)


    # make a prediction
    # row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
    # yhat = model.predict([row])
    # print('Predicted: %.3f' % yhat)