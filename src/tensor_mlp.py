# mlp for regression
from numpy import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras

# import sys
# sys.setrecursionlimit(15000)

if __name__ == '__main__':
    # load the dataset
    path = '../model/data.csv'
    df = pd.read_csv(path, sep='\t')
    # df.set_index(keys='api', inplace=True)
    df.drop('api', axis=1, inplace=True)
    print(df.head(5))
    # split into input and output columns
    X = df.values[:, :-1]
    y = df.values[:, -1]
    # split into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
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
    model.add(Dense(3, activation='relu', kernel_initializer=kernel_init, input_shape=(n_features,)))
    model.add(Dense(50, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(50, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(50, activation='relu', kernel_initializer=kernel_init))
    # model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    # model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    # model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    model.add(Dense(3))
    # compile the model
    # optimizer='adam'
    # optimizer='rmsprop'
    opt = keras.optimizers.Adam(learning_rate=0.005)
    model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
    # fit the model
    model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)
    # evaluate the model
    error = model.evaluate(X_test, y_test, steps=1, use_multiprocessing=True, verbose=1)
    print('THIS IS THE ERROR: ', error)
    print('MSE: %.3f, RMSE: %.3f' % (error[1], sqrt(error[1])))

    # y_true = predictions.select('label').toPandas()
    # y_pred = predictions.select('prediction').toPandas()
    # r2_score = sklearn.metrics.r2_score(y_true, y_pred)
    # print('r2_score: {:4.3f}'.format(r2_score))



    # make a prediction
    # row = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
    # yhat = model.predict([row])
    # print('Predicted: %.3f' % yhat)