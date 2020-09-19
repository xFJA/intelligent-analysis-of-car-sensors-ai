import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.callbacks import ModelCheckpoint
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .common import get_base64


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def predict(csv_file, feature, epochs):

    # Read file
    df = pd.read_csv(csv_file)
    print(feature)
    print(df)
    predict_column = df[feature]

    # TODO: Remove this step
    # Remove Unnamed column
    #df = df.drop(labels='Unnamed: 27', axis=1)

    # Apply PCA
    # Get features from the first row
    features = df.columns.values
    x = df.loc[:, features].values

    # Apply PCA
    # TODO: change labels for more components using function
    PCA_model = PCA(n_components=3)

    principal_components = PCA_model.fit_transform(x)
    principalDf = pd.DataFrame(
        data=principal_components, columns=['pc1', 'pc2', 'pc3'])
    principalDf['predict'] = pd.Series(predict_column)
    print(principalDf)
    df = principalDf

    # Create scaler to normalize features to [0-1]
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the dataset
    values = df.values.astype('float32')
    scaled_values = scaler.fit_transform(values)
    print(df)
    # Set duration to predict
    n_15min_interval = 4
    #n_features = df.shape[1]
    n_features = 4
    print('n_features: ', n_features)

    # Construct input records capturing more than 1 hour of input time steps
    df_reframed = series_to_supervised(scaled_values, n_15min_interval, 1)
    print(df_reframed.head())

    # Drop current time frame except the precitor
    df_reframed_columns = n_features*n_15min_interval

    print('df_reframed columns: ', df_reframed_columns)
    df_reframed.drop(df_reframed.columns[range(
        df_reframed_columns, df_reframed_columns+n_features-1)], axis=1, inplace=True)
    print('df_reframed after current time deletion:', df_reframed.head())
    # Split dataset into train set and test set
    n_obs = n_15min_interval * n_features
    print('n_obs: ', n_obs)
    n_train_records = int(len(df_reframed.values) * 0.7)
    print('n train records: ', n_train_records)
    train = df_reframed.values[:n_train_records, :]
    test = df_reframed.values[n_train_records:, :]
    print('Train shape:', train.shape)
    print('Test shape:', test.shape)

    # Construct X set as the features and Y set as the predictor/ or the output variable
    X_train, y_train = train[:, :n_obs], train[:, -1]
    X_test, y_test = test[:, :n_obs], test[:, -1]
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)
    print('\nX_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)

    # The inputs (X) are reshaped into the 3D format expected by RNN, namely (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], n_15min_interval, n_features))
    X_test = X_test.reshape((X_test.shape[0], n_15min_interval, n_features))
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # Design model: RNN based on hierarchically stacked three layered LSTM network consisting of 256, 128 and 64 LSTM Networks
    model = Sequential()
    model.add(LSTM(n_obs, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2])))
    #model.add(LSTM(80, return_sequences=True))
    # model.add(LSTM(40, return_sequences=True))
    # model.add(LSTM(20))
    model.add(LSTM(24, return_sequences=True))
    model.add(LSTM(12))
    model.add(Dense(1))

    # Compile model with Mean Squarred Error and ADAM optimizer
    model.compile(loss='mse', optimizer='adam')

    # Train model
    train_epochs = int(epochs)
    batch_size = 64
    validation_split = 0.3

    start = time.time()  # training start

    history = model.fit(X_train, y_train, epochs=train_epochs, batch_size=batch_size,
                        validation_split=validation_split, verbose=1, shuffle=False)

    end = time.time()
    print('Training time: ', (end-start))

    # Plot learning curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.savefig("learning_curve.png", bbox_inches='tight')
    learning_curve_plot = get_base64(plt, 'tight')
    plt.clf()

    yhat = model.predict(X_test, batch_size=batch_size)
    print('yhat shape: ', yhat.shape)

    X_test = X_test.reshape((X_test.shape[0], n_15min_interval * n_features))
    print('X_test shape: ', X_test.shape)

    mse = mean_squared_error(y_test, yhat)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    # Rescale predictions
    inv_yhat = np.concatenate((X_test[:, -(n_features-1):],yhat), axis=1)
    print(inv_yhat.shape)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, -1]

    # Rescale test data (y_test)
    y_test = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((X_test[:, -(n_features-1):], y_test), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, -1]

    # Plot prediction
    x_ticks_range = range(len(inv_y))
    print('x ticks range', len(inv_y))
    print(inv_y)
    plt.figure(figsize=(36, 7))
    plt.plot(x_ticks_range, inv_y, label='actual')
    plt.plot(x_ticks_range, inv_yhat, label='predicted')
    plt.xticks(x_ticks_range, rotation='vertical')
    plt.legend()
    plt.ylabel(feature+' (per 15min)')
    #plt.savefig("prediction.png", bbox_inches='tight')
    prediction_plot = get_base64(plt, 'tight')
    plt.clf()

    return (learning_curve_plot.decode('ascii'), prediction_plot.decode('ascii'), rmse, (end-start))
