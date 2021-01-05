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
    df.set_index(keys='api', inplace=True)
    # df.drop(df.columns[1], axis=1, inplace=True)
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
    # activation='linear'
    # activation='relu'
    # kernel_initializer='glorot_uniform'
    # kernel_initializer='he_uniform'
    # kernel_initializer='normal'
    # GlorotNormal
    model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal', input_shape=(n_features,)))
    model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    model.add(Dense(50, activation='relu', kernel_initializer='GlorotNormal'))
    # model.add(Dense(12, activation='relu', kernel_initializer='glorot_uniform'))
    # model.add(Dense(12, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(50))
    # compile the model
    # optimizer='adam'
    # optimizer='rmsprop'
    opt = keras.optimizers.Adamax(learning_rate=0.005)
    model.compile(optimizer='adamax', loss='mse', metrics=['mse', 'mae'])
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