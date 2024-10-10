import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

def save_predictions(current_prediction):
    if os.path.exists('prenas.txt'):
        # Leggi la previsione precedente e salvala in lastdax.txt
        with open('prenas.txt', 'r') as f:
            previous_prediction = f.read().strip()
        with open('lastnas.txt', 'w') as f:
            f.write(previous_prediction)
    # Salva la nuova previsione in predax.txt
    with open('prenas.txt', 'w') as f:
        f.write(str(current_prediction))

# Funzione per implementare l'ottimizzazione MFO (Moth Flame Optimization)
def moth_flame_optimization(num_iterations, num_moths, num_dimensions, lower_bound, upper_bound, fitness_function):
    # Inizializzazione delle posizioni delle falene (moths)
    moth_positions = np.random.uniform(lower_bound, upper_bound, (num_moths, num_dimensions))
    flame_positions = np.copy(moth_positions)
    flame_fitness = np.full(num_moths, np.inf)

    best_fitness = np.inf
    best_position = None

    for iteration in range(num_iterations):
        for i in range(num_moths):
            # Calcola il fitness per ogni falena
            fitness = fitness_function(moth_positions[i])
            if fitness < flame_fitness[i]:
                flame_fitness[i] = fitness
                flame_positions[i] = moth_positions[i]

            if fitness < best_fitness:
                best_fitness = fitness
                best_position = moth_positions[i]
        # Aggiornamento delle posizioni delle falene
        a = -1 + iteration * (-1 / num_iterations)
        for i in range(num_moths):
            distance_to_flame = np.abs(flame_positions[i] - moth_positions[i])
            t = np.random.uniform(-1, 1, size=(num_dimensions,))
            moth_positions[i] = flame_positions[i] - a * distance_to_flame * t
    return best_position, best_fitness

# Caricamento del dataset da CSV
def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    # Supponendo che il file CSV abbia colonne: Date, Open, High, Low, Close, Volume
    # Consideriamo solo le colonne OHLC e Volume come input
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    target = data['Close'].shift(-1)  # Prevediamo il prezzo di chiusura del giorno successivo

    # Pulizia dei dati (rimozione righe NaN dopo lo shift)
    features = features[:-1]
    target = target[:-1]

    # Normalizzazione dei dati
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    return train_test_split(features_scaled, target, test_size=0.2, shuffle=False)

# Funzione di fitness per MFO
def fitness_function(params):
    # Definiamo i parametri per il RandomForest
    max_depth = int(params[0])
    n_estimators = int(params[1])

    rf = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    return mse

# Parametri MFO
num_iterations = 50
num_moths = 10
num_dimensions = 2
lower_bound = [10, 100]  # Limiti per max_depth e n_estimators
upper_bound = [100, 1000]

# Carica il dataset
X_train, X_val, y_train, y_val = load_and_preprocess_data('nasdaq_historical_data.csv')

# Esegui MFO per ottimizzare i parametri di RandomForest
best_params, best_fitness = moth_flame_optimization(num_iterations, num_moths, num_dimensions, lower_bound, upper_bound, fitness_function)

# Addestramento finale del modello RandomForest con i parametri ottimizzati
best_max_depth = int(best_params[0])
best_n_estimators = int(best_params[1])

model = RandomForestRegressor(max_depth=best_max_depth, n_estimators=best_n_estimators)
model.fit(X_train, y_train)

# Previsione per il giorno successivo
last_day_features = X_val[-1].reshape(1, -1)
predicted_close_price = model.predict(last_day_features)

# Salva la previsione in una variabile chiamata 'predas'
predas = round(predicted_close_price[0], 2)
save_predictions(predas)
