import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
from tensorflow.keras import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU

def basic_analysis(dataset):
    # Load dataset
    data = pd.read_csv(dataset)
    # Choose the basic features from the dataset
    data = data[['Date', 'Open', 'High', 'Low', 'Close']]
    # Align features with target
    data['targ'] = data['Close'].shift(-1)
    # Interpolate missing values
    data = data.interpolate(method='linear')
    # Drop NA values
    data.dropna(axis=0, inplace=True)

    # Define X and y as arrays
    X = data[['Open', 'High', 'Low', 'Close']]
    y = data['targ']

    # Create train-test split
    split = int(len(data)*0.8)
    valspl = int(len(data)*0.6)

    # Large train
    X_tr = X[:split].to_numpy()
    y_tr = y[:split].to_numpy()

    # Val-train
    X_train = X[:valspl].to_numpy()
    y_train = y[:valspl].to_numpy()

    # Validation
    X_val = X[valspl:split].to_numpy()
    y_val = y[valspl:split].to_numpy()

    # Test
    X_test = X[split:].to_numpy()
    y_test = y[split:].to_numpy()

    # Normalizing the data
    minimum = np.min(X_train, axis=0)
    maximum = np.max(X_train, axis=0)

    # First use
    X_train = (X_train - minimum) / (maximum - minimum)
    X_val = (X_val - minimum) / (maximum - minimum)

    min2 = np.min(X_tr, axis=0)
    max2 = np.max(X_tr, axis=0)

    # Later use
    X_tr = (X_tr - min2) / (max2 - min2)
    X_test = (X_test - min2) / (max2 - min2)

    # Reshaping the data
    X_train_series = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_val_series = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

    # Later use
    X_tr_series = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
    X_test_series = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Define different algorithms
    models = [SimpleRNN, LSTM, GRU]

    # Build function to create model architecture that enables hyperparameter tuning
    def build_model(mod, units):
        model = Sequential()
        model.add(mod(units, input_shape=((X_train_series.shape[1], X_train_series.shape[2])), activation='tanh', return_sequences=False))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
        return model

    # Initialize array for errors
    maes = [[],[],[]]
    
    # Test different models
    for mod in range(len(models)):
        # Test different sizes of hidden layer
        for i in range(2, 15):
            model = build_model(models[mod], i)
            model.fit(X_train_series, y_train, epochs = 100, verbose=2)
            pred = model.predict(X_val_series)
            maes[mod].append(mean_absolute_error(y_val, pred))

    best = np.argmin(maes, axis = 1) + 2

    finmae = []
    finmape = []
    for mod in range(len(models)):
        model = build_model(models[mod], best[mod])
        model.fit(X_tr_series, y_tr, epochs = 100, verbose=2)
        pred = model.predict(X_test_series)
        finmae.append(mean_absolute_error(y_test, pred))
        finmape.append(mean_absolute_percentage_error(y_test, pred))

    return finmae, finmape, best

euro = basic_analysis('Euro_data.csv')
np.savetxt('Basic analysis results/euro_lag1.txt', euro)
jpy = basic_analysis('JPY_data.csv')
np.savetxt('Basic analysis results/jpy_lag1.txt', jpy)
gbp = basic_analysis('GBP_data.csv')
np.savetxt('Basic analysis results/gbp_lag1.txt', gbp)
cny = basic_analysis('CNY_data.csv')
np.savetxt('Basic analysis results/cny_lag1.txt', cny)

print('Euro results')
print(euro)
print('\n')
print('JPY results')
print(jpy)
print('\n')
print('GBP results')
print(gbp)
print('\n')
print('CNY results')
print(cny)