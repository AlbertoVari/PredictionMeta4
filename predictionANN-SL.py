import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Funzione per ottenere il valore di chiusura dall'ultima riga del file
def get_last_close_from_csv(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    if not data.empty:
        return data['Close'].iloc[-1]
    else:
        raise Exception("The file is empty or the specified column does not exist")

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
residuals = test[look_back:] - scaler.transform(linear_pred_scaled)

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


file_sorgente = "predax.txt"
file_destinazione = "lastdax.txt"

with open(file_sorgente, "r") as file_sorgente:
    contenuto = file_sorgente.read()

with open(file_destinazione, "w") as file_destinazione:
    file_destinazione.write(contenuto)



print("Today DAX (GDAXI) prediction : ", combined_pred[0])

# Supervised learning: updating the model with actual values
# Ottieni il valore di chiusura dall'ultima riga del file csv
actual_value = get_last_close_from_csv('dax_historical_data.csv')

# Scrivi il valore nel file actual.txt
with open("actual.txt", "w") as file:
    file.write(str(actual_value))

# Aggiornare i dati di addestramento con il nuovo valore
actual_value_scaled = scaler.transform([[actual_value]])[0][0]
new_X_train = np.append(X_train, [X_test[-1]], axis=0)
new_Y_train = np.append(Y_train, [actual_value_scaled], axis=0)

# Update linear regression model
regressor.fit(new_X_train, new_Y_train)
linear_pred = regressor.predict(X_test)
linear_pred = linear_pred.reshape(-1, 1)
linear_pred_scaled = scaler.inverse_transform(linear_pred)

# Calculate new residuals
residuals = test[look_back:] - scaler.transform(linear_pred_scaled)

# Update ANN
model.fit(new_X_train, new_Y_train, epochs=epoche, batch_size=batch_size, verbose=2)

# Previsioni ANN aggiornate
ann_pred = model.predict(X_test_ann)
ann_pred = scaler.inverse_transform(ann_pred)

# Combinare i risultati aggiornati
combined_pred = linear_pred_scaled + ann_pred
print("Updated DAX (GDAXI) prediction : ", combined_pred[0])
str_combined = str(combined_pred[0])
with open("predax.txt", 'w') as file:
        # Scrive il valore nel file
        file.write(str_combined)

# Plot dei risultati
plt.figure(figsize=(12, 6))

# Plot the 'True Value' series
plt.plot(series[-len(test):], label='True Value')

# Plot the 'Linear Regression' series
linear_pred = scaler.inverse_transform(linear_pred_scaled)
plt.plot(linear_pred, label='Linear Regression')

# Plot the 'Combined Linear Regression + ANN' series
combined_pred = linear_pred + ann_pred
plt.plot(combined_pred, label='Combined Linear Regression + ANN')

plt.legend()
plt.savefig('dax-pred.png')  # Save the figure to the specified path
plt.show()
