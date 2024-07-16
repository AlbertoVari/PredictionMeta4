import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 1. Caricare e preprocessare i dati
data = pd.read_csv('dax_historical_data.csv', index_col='Date', parse_dates=True)
series = data['Close'].values

# Normalizzare i dati
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_series = scaler.fit_transform(series.reshape(-1, 1))

# Suddividere i dati in training e test set
train_size = int(len(scaled_series) * 0.8)
train, test = scaled_series[:train_size], scaled_series[train_size:]

# 2. Applicare Regressione Multipla
def create_features(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 3
X_train, Y_train = create_features(train, look_back)
X_test, Y_test = create_features(test, look_back)

# Regressione Lineare Multipla
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
linear_pred = regressor.predict(X_test)
linear_pred = linear_pred.reshape(-1, 1)
linear_pred_scaled = scaler.inverse_transform(linear_pred)

# Calcolare i residui
residuals = scaler.transform(test[look_back:].reshape(-1, 1)) - linear_pred_scaled

# 3. Costruire e addestrare il modello ANN
# Rimodellare input per [samples, features]
X_train_ann, Y_train_ann = X_train, Y_train
X_test_ann, Y_test_ann = X_test, residuals

# Parametri modificabili
neuroni_per_livello = 50
numero_livelli_nascosti = 2
funzione_attivazione = 'relu'
dropout_rate = 0.2
learning_rate = 0.001
epoche = 50
batch_size = 32

#Note sui Parametri Modificabili
#   neuroni_per_livello: Numero di neuroni in ogni livello nascosto.
#   numero_livelli_nascosti: Numero di livelli nascosti nella rete.
#   funzione_attivazione: Funzione di attivazione utilizzata nei livelli nascosti.
#   dropout_rate: Tasso di dropout per prevenire l'overfitting.
#   learning_rate: Tasso di apprendimento per l'ottimizzatore.
#   epoche: Numero di epoche di addestramento.
#   batch_size: Dimensione del batch per l'addestramento.
#


# Costruire il modello ANN
model = Sequential()
model.add(Input(shape=(look_back,)))
for _ in range(numero_livelli_nascosti):
    model.add(Dense(neuroni_per_livello, activation=funzione_attivazione))
    model.add(Dropout(dropout_rate))
model.add(Dense(1))
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(X_train_ann, Y_train_ann, epochs=epoche, batch_size=batch_size, verbose=2)

# Previsioni ANN
ann_pred = model.predict(X_test_ann)
ann_pred = scaler.inverse_transform(ann_pred)

# 4. Combinare i risultati di Regressione Multipla e ANN
combined_pred = linear_pred_scaled + ann_pred
print("Today DAX (GDAXI) prediction : ",combined_pred[0])

# Plot dei risultati
#plt.figure(figsize=(12, 6))
#plt.plot(series[-len(test):], label='True Value')
#plt.plot(scaler.inverse_transform(linear_pred_scaled), label='Linear Regression')
#plt.plot(scaler.inverse_transform(linear_pred_scaled) + ann_pred, label='Combined Linear Regression + ANN')
#plt.legend()
#plt.show()




