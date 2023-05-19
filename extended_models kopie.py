import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
from tensorflow.keras import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU

def extended_analysis(dataset, optimal):
    # Load dataset
    data = pd.read_csv(dataset)
    # Choose the extended features from the dataset
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
              'momentum_4', 'momentum_5', 'momentum_6', 'momentum_7', 'momentum_8', 'momentum_9','momentum_10']]
    y = data['targ']

    # Create train-test split
    split = int(len(data)*0.8)
    valspl = int(len(data)*0.6)

    # Large train
    X_tr = X[:split].to_numpy()
    y_tr = y[:split].to_numpy()

    # Test
    X_test = X[split:].to_numpy()
    y_test = y[split:].to_numpy()

    min2 = np.min(X_tr, axis=0)
    max2 = np.max(X_tr, axis=0)

    # Later use
    X_tr = (X_tr - min2) / (max2 - min2)
    X_test = (X_test - min2) / (max2 - min2)

    # # Reshaping the data
    X_tr_series = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
    X_test_series = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # Define different algorithms
    models = [SimpleRNN, LSTM, GRU]

    # Build function to create model architecture that enables hyperparameter tuning
    def build_model(mod, units):
        model = Sequential()
        model.add(mod(units, input_shape=((X_tr_series.shape[1], X_tr_series.shape[2])), activation='tanh', return_sequences=False))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
        return model

    best = optimal

    finmae = []
    finmape = []
    for mod in range(len(models)):
        model = build_model(models[mod], best[mod])
        model.fit(X_tr_series, y_tr, epochs = 100, verbose=2)
        pred = model.predict(X_test_series)
        finmae.append(mean_absolute_error(y_test, pred))
        finmape.append(mean_absolute_percentage_error(y_test, pred))

    return finmae, finmape, best

eur = extended_analysis('Euro_data.csv', [6,13,9])
np.savetxt('Extended analysis results/eur_ext_1.txt', eur)
gbp = extended_analysis('GBP_data.csv', [8,8,2])
np.savetxt('Extended analysis results/gbp_ext_1.txt', gbp)
jpy = extended_analysis('JPY_data.csv', [14,14,14])
np.savetxt('Extended analysis results/jpy_ext_1.txt', jpy)
cny = extended_analysis('CNY_data.csv', [14,14,14])
np.savetxt('Extended analysis results/cny_ext_1.txt', cny)