#
# Quantum Machine Learning Intraday DAX Trading Algorithm
'''
Overview

This Python script implements an intraday trading strategy for the DAX index
using Quantum Machine Learning (QML) techniques via the Qiskit framework.
 The algorithm retrieves market data, applies quantum feature mapping and ansatz,
and uses quantum computations to predict market movements and execute trades.

'''
import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import feedparser
import matplotlib.pyplot as plt


# Quantum Feature
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterExpression
import torch
import numpy as np
import traceback

estimator, qc, observable, ansatz = None, None, None, None


# Creazione della mappa delle feature quantistiche
def quantum_feature_map():
    return ZZFeatureMap(feature_dimension=2, reps=2)

# Creazione dell'ansatz
def quantum_ansatz():
    return RealAmplitudes(num_qubits=2, reps=2)

# Modello quantistico aggiornato per Qiskit 1.2+
def quantum_model():
    print("Inizializzazione del modello quantistico...")
    try:
        feature_map = quantum_feature_map()
        ansatz = quantum_ansatz()
        print("Feature map e ansatz creati.")

        qc = QuantumCircuit(2)
        qc.compose(feature_map, inplace=True)
        qc.compose(ansatz, inplace=True)
        print("Circuito quantistico creato.")
        fig = qc.draw(output="mpl")
        plt.savefig("circuito_quantistico.png", dpi=300) 

        estimator = StatevectorEstimator()
        if estimator is None:
            raise ValueError("Errore: lo StatevectorEstimator non è stato inizializzato correttamente.")
        print("StatevectorEstimator inizializzato.")

        try:
            observable = SparsePauliOp(["ZZ", "XX"], coeffs=[1, 1])
            print("Osservabile creato correttamente.")
        except Exception as e:
            print(f"Errore nella creazione dell'osservabile: {e}")
            traceback.print_exc()
            return None
        return estimator, qc, observable, ansatz
    except Exception as e:
        print(f"Errore in quantum_model(): {e}")
        traceback.print_exc()
        return None, None, None, None  # ✅ Evita che una funzione fallisca senza valori

# Funzione per eseguire il circuito quantistico
def quantum_forward(input_data, weights):
    global estimator, qc, observable, ansatz  # Dichiarazione globale PRIMA di usarli
    try:
        # Se weights è un ParameterView, convertiamolo in numeri reali casuali
        if any(isinstance(w, ParameterExpression) for w in weights):
            print("⚠️ Attenzione: weights contiene ancora parametri simbolici! Sostituzione con numeri reali casuali.")
            weights = np.random.rand(len(weights))  # Sostituiamo con numeri reali
        weights_resized = np.array(weights, dtype=np.float64)  # ✅ Ora sono numeri reali


        input_array = input_data.detach().cpu().numpy().flatten().astype(np.float64)
        param_values = np.array(list(input_array) + list(weights_resized), dtype=np.float64).flatten()

        param_keys = list(qc.parameters)

        if len(param_keys) != len(param_values):
            raise ValueError("Mismatch tra parametri attesi e valori forniti!")

        param_dict = {k: float(v) for k, v in zip(param_keys, param_values)}
        bound_circuit = qc.assign_parameters(param_dict)

        job = estimator.run([(bound_circuit, observable)])
        result = job.result()

        expectation_value = result[0].data.evs.real  # Assicura che il valore sia reale  # ✅ Converte in numero Python

        # ✅ Converti in tensore PyTorch senza errori
        return torch.tensor([float(expectation_value)], dtype=torch.float32)

    except Exception as e:
        print(f"Errore in quantum_forward(): {e}")
        traceback.print_exc()
        return None


estimator, qc, observable, ansatz = quantum_model()
print("QNN Model:", estimator, qc, observable, ansatz)
print()


# Funzione per decidere il trade
def decide_trade_quantum(data):
    global ENTRY_PRICE, TRADE_TYPE
    latest_price = data['Close'].iloc[-1]
    sma_short = data['SMA_Short'].iloc[-1]
    sma_long = data['SMA_Long'].iloc[-1]

    input_data = torch.Tensor([[sma_short, sma_long]])
    if ansatz is None:
         raise ValueError("Errore: il modello quantistico non è stato inizializzato correttamente.")

    random_weights = np.random.rand(len(ansatz.parameters))  # 🔹 Genera pesi casuali inizializzati

    prediction = quantum_forward(input_data, random_weights)  # 🔹 Passiamo numeri reali

    if prediction is None:
        raise ValueError("Errore: la previsione del modello quantistico ha restituito None.")

    prediction = prediction.item()
    new_decision = "BUY" if prediction > 0 else "SELL"

    if new_decision != TRADE_TYPE:
        ENTRY_PRICE = latest_price
        TRADE_TYPE = new_decision

    return new_decision

# Parametri principali
SYMBOL = "^GDAXI"
TIMEFRAME = "1m"
ENTRY_PRICE = float(input('Inserisci il prezzo di entrata: '))
TRADE_TYPE = input('Inserisci il tipo di contratto (BUY o SELL): ').strip().upper()
POSITION_SIZE = float(input('Inserisci la quota del contratto: '))
TAKE_PROFIT = float(input('Inserisci il Take Profit in €: '))
STOP_LOSS = float(input('Inserisci lo Stop Loss in €: '))

# Credenziali e configurazione email
EMAIL_SENDER = ""
EMAIL_PASSWORD = ""
EMAIL_RECEIVER = ""
SMTP_SERVER = ""
SMTP_PORT =  XXX

# Loop principale per il trading intraday
def main():
    while True:
        try:
            data = yf.download(SYMBOL, period="1d", interval=TIMEFRAME)
            data['SMA_Short'] = data['Close'].rolling(window=5).mean()
            data['SMA_Long'] = data['Close'].rolling(window=20).mean()
            decision = decide_trade_quantum(data)
            now = datetime.now()
            print(f"Nuova decisione (Quantum): {decision}"," ",now)

            # Creazione del messaggio email
            msg = EmailMessage()
            msg["Subject"] = "Intraday DAX"
            msg["From"] = EMAIL_SENDER
            msg["To"] = EMAIL_RECEIVER
            msg.set_content(f"""
            Nuova decisione (Quantum): {decision}
            Data e ora: {now.strftime("%Y-%m-%d %H:%M:%S")}
            """)

            # Connessione e invio email
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)

            time.sleep(60)

        except Exception as e:
            print(f"Errore: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()
