import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from pennylane import grad


# Scarica i dati storici del DAX
ticker = "^GDAXI"
dax_data = yf.Ticker(ticker)
hist = dax_data.history(period="max")

# Filtra i dati storici dal 2015 in poi
hist = hist[hist.index >= "2017-01-01"]

# Salva i dati filtrati in un file CSV
hist.to_csv('dax_historical_data.csv')


# Supponiamo di avere un file CSV con i dati storici
data = pd.read_csv('dax_historical_data.csv')
prices = data['Close'].values

# Normalizzazione dei dati
prices_normalized = (prices - np.mean(prices)) / np.std(prices)


# Econding con Angle Encoding

def angle_encoding(x):
    import pennylane as qml
    n_qubits = len(x)
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit():
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit()

# Definzione modello quantistico

n_qubits = 4  # Numero di qubit (dipende da come encodi i tuoi dati)
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_model(params, x):
    # Encoding dei dati
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)

    # Circuito variazionale

    qml.templates.layers.StronglyEntanglingLayers(params, wires=range(n_qubits))

    return qml.expval(qml.PauliZ(0))

# Preparazione addestramento

X = []
Y = []

# Creazione delle sequenze temporali
window_size = n_qubits
for i in range(len(prices_normalized) - window_size):
    X.append(prices_normalized[i:i+window_size])
    Y.append(prices_normalized[i+window_size])

X = np.array(X)
Y = np.array(Y)

# Addestramento Modello
# Inizializzazione dei parametri
np.random.seed(0)
params = 0.01 * np.random.randn(3, n_qubits, 3, requires_grad=True)

def cost(params, X, Y):
    predictions = [quantum_model(params, x) for x in X]
    predictions = np.array(predictions, requires_grad=True)
    return np.mean((predictions - Y)**2)


# Scegli un input dal tuo dataset
sample_input = X[0]

# Calcola il gradiente rispetto agli input
gradient_fn = qml.grad(quantum_model, argnum=1)
gradients = gradient_fn(params, sample_input)

print("Gradiente rispetto agli input:", gradients)

# Addstramento Modello
print(qml.draw(quantum_model)(params, X[0]))


opt = qml.AdamOptimizer(stepsize=0.5)
epochs = 50
for epoch in range(epochs):
    params, prev_cost = opt.step_and_cost(lambda v: cost(v, X, Y), params)
    print(f'Epoch {epoch+1}: Params = {params}')
    mse = cost(params, X, Y)
    print(f'Epoch {epoch+1}: MSE = {mse}')

# Valutazione modello
# Previsioni
predictions = [quantum_model(params, x=x) for x in X]

# Generazione del grafico
plt.plot(Y, label='Valori Reali')
plt.plot(predictions, label='Previsioni')
plt.legend()

# Salvataggio del grafico in un file PNG
plt.savefig('dax30_previsioni.png')

# Chiudere la figura per liberare memoria
plt.close()


