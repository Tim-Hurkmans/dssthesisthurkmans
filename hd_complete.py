import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
from tensorflow.keras import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU
import tensorflow as tf


def higher_dimensions(dataset, lags, optimal, modelbas):
    # Load dataset
    data = pd.read_csv(dataset)
    # Choose extended features
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'cpiret', 'FF_O', 'b1ret']]
    # Create momentum
    data['return'] = data['Close'].pct_change()
    for i in range(1, 11):
        momentum = data['return'].rolling(window=i).sum()
        data[f'momentum_{i}'] = momentum.shift(1)
    # Align features with target
    data['targ'] = data['Close'].shift(-1)
    # Interpolate missing values
    data = data.interpolate(method='linear')
    # Drop NA values
    data.dropna(axis=0, inplace=True)
    # Define X and y as arrays
    X = data[['Open', 'High', 'Low', 'Close', 'cpiret', 'FF_O', 'b1ret', 'momentum_1', 'momentum_2','momentum_3',
              'momentum_4', 'momentum_5', 'momentum_6', 'momentum_7', 'momentum_8', 'momentum_9','momentum_10']].values
    y = data['targ'].values

    days = lags
    features = X.shape[1]
    samples = X.shape[0] - days + 1

    rolling_window = np.zeros((samples, days, features))
    for i in range(samples):
      rolling_window[i] = X[i:i+days]

    X = rolling_window
    y = y[days-1:]

    # Create train-test split
    split = int(X.shape[0]*0.8)

    # Large train
    X_tr = X[:split]
    y_tr = y[:split]

    # Test
    X_test = X[split:]
    y_test = y[split:]

    min2 = np.min(X_tr, axis=0)
    max2 = np.max(X_tr, axis=0)

    # Later use
    X_tr = (X_tr - min2) / (max2 - min2)
    X_test = (X_test - min2) / (max2 - min2)

    # Later use
    X_tr_series = np.reshape(X_tr, (X_tr.shape[0], X_tr.shape[1], X_tr.shape[2]))
    X_test_series = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_tr.shape[2]))

    # Build function to create model architecture that enables hyperparameter tuning
    def build_model(mod, units):
        model = Sequential()
        model.add(mod(units, input_shape=((X_tr_series.shape[1], X_tr_series.shape[2])), activation='tanh', return_sequences=False))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
        return model

    best = optimal
    
    model = build_model(modelbas, best)
    model.fit(X_tr_series, y_tr, epochs = 100, verbose = 2)
    pred = model.predict(X_test_series)
    finmae = mean_absolute_error(y_test, pred)
    finmape = mean_absolute_percentage_error(y_test, pred)

    return [finmae, finmape]


# Euro HD models models
eur_srnn = []
eur_lstm = []
eur_gru = []

for i in range(2,11):
   eur_srnn.append(higher_dimensions('Euro_data.csv', i, 5, SimpleRNN))
for i in range(2,11):
   eur_lstm.append(higher_dimensions('Euro_data.csv', i, 5, LSTM))
for i in range(2,11):
   eur_gru.append(higher_dimensions('Euro_data.csv', i, 8, GRU))

euro = np.array([np.array(eur_srnn).T,
                 np.array(eur_lstm).T ,
                 np.array(eur_gru).T])
np.save('HD incl mape ext/eur_to_ext.npy', euro)

# GBP HD models
gbp_srnn = []
gbp_lstm = []
gbp_gru = []

for i in range(2,11):
   gbp_srnn.append(higher_dimensions('GBP_data.csv', i, 10, SimpleRNN))
for i in range(2,11):
   gbp_lstm.append(higher_dimensions('GBP_data.csv', i, 4, LSTM))
for i in range(2,11):
   gbp_gru.append(higher_dimensions('GBP_data.csv', i, 3, GRU))

gbp = np.array([np.array(gbp_srnn).T,
                 np.array(gbp_lstm).T,
                 np.array(gbp_gru).T])
np.save('HD incl mape ext/gbp_to_ext.npy', gbp)

# JPY HD models models
jpy_srnn = []
jpy_lstm = []
jpy_gru = []

for i in range(2,11):
   jpy_srnn.append(higher_dimensions('JPY_data.csv', i, 14, SimpleRNN))
for i in range(2,11):
   jpy_lstm.append(higher_dimensions('JPY_data.csv', i, 14, LSTM))
for i in range(2,11):
   jpy_gru.append(higher_dimensions('JPY_data.csv', i, 14, GRU))

jpy = np.array([np.array(jpy_srnn).T,
                 np.array(jpy_lstm).T ,
                 np.array(jpy_gru).T])
np.save('HD incl mape ext/jpy_to_ext.npy', jpy)

# CNY HD models
cny_srnn = []
cny_lstm = []
cny_gru = []

for i in range(2,11):
   cny_srnn.append(higher_dimensions('CNY_data.csv', i, 10, SimpleRNN))
for i in range(2,11):
   cny_lstm.append(higher_dimensions('CNY_data.csv', i, 14, LSTM))
for i in range(2,11):
   cny_gru.append(higher_dimensions('CNY_data.csv', i, 14, GRU))

cny = np.array([np.array(cny_srnn).T,
                 np.array(cny_lstm).T,
                 np.array(cny_gru).T])
np.save('HD incl mape ext/cny_to_ext.npy', cny)