import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pandas_ta as ta  # Libreria per l'analisi tecnica

# Funzione per salvare la previsione attuale e quella precedente
def save_predictions(current_prediction):
    if os.path.exists('predax.txt'):
        # Leggi la previsione precedente e salvala in lastdax.txt
        with open('predax.txt', 'r') as f:
            previous_prediction = f.read().strip()
        with open('lastdax.txt', 'w') as f:
            f.write(previous_prediction)
    # Salva la nuova previsione in predax.txt
    with open('predax.txt', 'w') as f:
        f.write(str(current_prediction))

# Caricare e preprocessare i dati
data = pd.read_csv('dax_historical_data.csv', index_col='Date', parse_dates=True)

# Assicurarsi che l'indice sia un DatetimeIndex senza fuso orario
# data.index = data.index.tz_localize(None)

# Utilizzare solo i dati dal 2016 in poi
#start_date = pd.Timestamp('2016-01-01')
# data = data[data.index >= start_date]

# Aggiungere indicatori tecnici
data['RSI'] = ta.rsi(data['Close'], length=14)
macd = ta.macd(data['Close'])
data['MACD'] = macd['MACD_12_26_9']
data['MACD_signal'] = macd['MACDs_12_26_9']
data['MA50'] = ta.sma(data['Close'], length=50)
data['MA200'] = ta.sma(data['Close'], length=200)

# Rimuovere le righe con valori NaN
data = data.dropna()

series = data['Close'].values
features = data[['RSI', 'MACD', 'MACD_signal', 'MA50', 'MA200']].values

# Normalizzare i dati
scaler_series = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler_series.fit_transform(series.reshape(-1, 1))

scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(features)

# Suddividere i dati in training e test set
train_size = int(len(scaled_series) * 0.8)
train_series, test_series = scaled_series[:train_size], scaled_series[train_size:]
train_features, test_features = scaled_features[:train_size], scaled_features[train_size:]

# Applicare Regressione Multipla
def create_features(dataset_series, dataset_features, look_back=1):
    X, Y = [], []
    for i in range(len(dataset_series) - look_back):
        X.append(np.hstack((dataset_series[i:(i + look_back), 0], dataset_features[i + look_back, :])))
        Y.append(dataset_series[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, Y_train = create_features(train_series, train_features, look_back)
X_test, Y_test = create_features(test_series, test_features, look_back)

# Estrarre il valore di chiusura attuale da Yahoo Finance
ticker = "^GDAXI"
dax_data = yf.Ticker(ticker)
current_close = dax_data.history(period="1d")['Close'].values[0]

# Aggiungere il valore di chiusura attuale come feature
current_close_scaled = scaler_series.transform([[current_close]])[0][0]

X_train = np.hstack((X_train, np.full((X_train.shape[0], 1), current_close_scaled)))
X_test = np.hstack((X_test, np.full((X_test.shape[0], 1), current_close_scaled)))

# Regressione Lineare Multipla
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
linear_pred = regressor.predict(X_test)
linear_pred = linear_pred.reshape(-1, 1)

# Costruire e addestrare il modello ANN
X_train_ann, Y_train_ann = create_features(train_series, train_features, look_back)
X_test_ann, Y_test_ann = create_features(test_series, test_features, look_back)

X_train_ann = np.hstack((X_train_ann, np.full((X_train_ann.shape[0], 1), current_close_scaled)))
X_test_ann = np.hstack((X_test_ann, np.full((X_test_ann.shape[0], 1), current_close_scaled)))

def build_ann_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

X_train_ann = np.reshape(X_train_ann, (X_train_ann.shape[0], X_train_ann.shape[1]))
X_test_ann = np.reshape(X_test_ann, (X_test_ann.shape[0], X_test_ann.shape[1]))

ann_model = build_ann_model((X_train_ann.shape[1],))
ann_model.fit(X_train_ann, Y_train_ann, epochs=100, batch_size=64, verbose=1)

ann_pred = ann_model.predict(X_test_ann)
ann_pred = ann_pred.reshape(-1, 1)

# Denormalizzazione
linear_pred_scaled = scaler_series.inverse_transform(linear_pred)
ann_pred_scaled = scaler_series.inverse_transform(ann_pred)

# Combinazione delle previsioni
combined_pred = 0.5 * (linear_pred_scaled + ann_pred_scaled)

current_prediction = combined_pred[-1][0]
save_predictions(current_prediction)

rmse = np.sqrt(mean_squared_error(Y_test, combined_pred))
r2 = r2_score(Y_test, combined_pred)

print(f'RMSE: {rmse}')
print(f'R^2: {r2}')

plt.figure(figsize=(14, 7))
plt.plot(data.index[-len(Y_test):], scaler_series.inverse_transform(Y_test.reshape(-1, 1)), label='Actual')
plt.plot(data.index[-len(Y_test):], combined_pred, label='Predicted')
plt.title('Actual vs Predicted Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('predicted_vs_actual.png')
plt.show()

