from numpy import sqrt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    path = '../model/data.csv'
    df = pd.read_csv(path, sep='\t', index_col=False)
    df.drop('api', axis=1, inplace=True)

    X = df.values[:, 1:-1]
    X_f = df.columns
    all_features = X_f[1:-1]
    y = df.values[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=88)

    print('shape of splits: ', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    n_features = X_train.shape[1]
    print('N FEATURES: ', n_features)

    model = Sequential()
    kernel_init = keras.initializers.GlorotUniform(seed=88)
    model.add(Dense(8, activation='relu', kernel_initializer=kernel_init, input_shape=(n_features,)))
    model.add(Dense(500, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(100, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(75, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(50, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(25, activation='relu', kernel_initializer=kernel_init))
    model.add(Dense(1))

    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer='adam', loss='mse', metrics='mse')
    model.fit(X_train, y_train, epochs=2, batch_size=2, validation_split=0.4, verbose=1, shuffle=False)

    error = model.evaluate(X_test, y_test, steps=1, use_multiprocessing=True, verbose=1)
    print('THIS IS THE ERROR: ', error)
    print('MSE: %.2f, RMSE: %.2f' % (error[0], sqrt(error[0])))
    
    X_p = df.iloc[:, 1:-2]
    print(X_p.head())
    y_p = df.loc[:, 'day365']
    print(y_p.head())
    mlp_model = KerasRegressor(build_fn=baseline_model, 
                               nb_epoch=100, 
                               batch_size=32, 
                               verbose=False)

    print('PASSED KERAS REGRESSOR')
    mlp_model.fit(X_p, y_p)
    print('PASSED MODEL . FIT COMMAND')
    perm = PermutationImportance(mlp_model, random_state=88)
    print(perm)
    print('PASSED PERMUTATION IMPORTANCE')

    feature_names = df.columns.tolist()[1:-1]
    eli5.show_weights(perm, target_names=df.columns.tolist()[1:-1], feature_names=df.columns.tolist()[1:-1])
    print('PASSED WEIGHTS')

