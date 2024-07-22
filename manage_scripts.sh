#!/bin/bash

# Percorso all'ambiente virtuale e ai tuoi script Python
VENV_DIR="/home/italygourmet_co/testapi/env"
SCRIPT_DIR="/home/italygourmet_co/testapi"

# Attiva l'ambiente virtuale
source $VENV_DIR/bin/activate

# Ferma i programmi capture.py, predictionANN.py e view_data_pre.py se sono in esecuzione
pkill -f capture.py
pkill -f predictionANN.py
pkill -f view_data_pre.py

# Avvia il programma capture.py
python3 $SCRIPT_DIR/capture.py &

# Attendi che capture.py sia avviato (puoi adattare il tempo di attesa se necessario)
sleep 10

# Avvia il programma predictionANN.py
python3 $SCRIPT_DIR/predictionANN.py &

# Attendi che predictionANN.py sia avviato (puoi adattare il tempo di attesa se necessario)
sleep 10

# Avvia il programma view_data_pre.py
python3 $SCRIPT_DIR/view_data_pre.py &

# Disattiva l'ambiente virtuale (opzionale, poiché il processo termina alla fine dello script)
deactivate