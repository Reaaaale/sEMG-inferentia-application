import threading
import queue
import time
import numpy as np
import torch
from collections import deque
import pandas as pd
import serial  
from src.model import Network
from config.config import FILTER_CONFIG, MODEL_CONFIG, WINDOW_CONFIG, SERIAL_CONFIG
from src.functions import (
    decode_data,
    load_model,
    preprocess_data,
    run_inference
)
"""Flusso
1. Acquisizione:
   - I dati raw vengono letti dalla porta seriale, decodificati e salvati in `shift_buffer`.
   - Quando `shift_buffer` è pieno, i dati vengono trasferiti a `buffer`.

2. Pre-elaborazione:
   - Quando `buffer` è pieno, si crea un batch preprocessato con `preprocess_data`.
   - Il batch preprocessato, i dati grezzi e un timestamp vengono aggiunti alla coda `data_queue`.

3. Inferenza:
   - I dati preprocessati vengono estratti da `data_queue` per l' inferenza.
   - Se data_queue supera la dimensione massima (`max_queue_size`), i batch più vecchi vengono scartati.
   - Viene elaborato solo il batch più recente per ridurre il ritardo tra acquisizione e inferenza.
   - Si calcolano ritardo (delay) e frequenza di predizione.

"""

#configurazione della dimensione dei buufer (uso gli stessi valori di finestra e shift usati in processing)
win_size_samples = int(WINDOW_CONFIG['win_size'] * FILTER_CONFIG['fs'])
win_shift_samples = int(WINDOW_CONFIG['win_shift'] * FILTER_CONFIG['fs'])
data_queue = queue.Queue()   # Coda per trasferire dati preprocessati e raw al thread di inferenza, prendo i dati preprocessati per passarli al thread di inferenza

# Buffer per l'acquisizione dati
buffer = deque(maxlen=win_size_samples)
raw_buffer = deque(maxlen=win_size_samples)   # Buffer parallelo per i dati raw, mi serve per debug così capisco quali valori vengono usati per quella predizione

# Array per salvare i dati
raw_data_list = []  # Per i dati raw
processed_data_list = []  # Per i dati preprocessati

# Variabile di controllo per fermare i thread
stop_event = threading.Event()

# Inizializzazione della porta seriale
try:
    ser = serial.Serial(SERIAL_CONFIG['port'], SERIAL_CONFIG['baud_rate'])
    print(f"Porta seriale {SERIAL_CONFIG['port']} aperta con baud rate {SERIAL_CONFIG['baud_rate']}.")
except serial.SerialException as e:
    print(f"Errore nell'apertura della porta seriale: {e}")
    exit()

# Thread di acquisizione e processing
def acquisition_thread():
    shift_buffer = deque(maxlen=win_shift_samples) #Creo un nuovo buffer per i nuovi campioni in modo da simulare lo shift della finestra
    while not stop_event.is_set():
        try:
            raw_data = ser.read(4)  # Legge 4 byte dalla porta seriale
            decoded_data = decode_data(raw_data)
            if decoded_data:
                dati_time = (decoded_data, time.time())  # Aggiungi un timestamp al dato, mi serve per calcolare il ritardo tra acquisizione e inferenza
                raw_data_list.append(dati_time)
                shift_buffer.append(dati_time)  # Aggiungi i dati con timestamp al buffer di shift
                raw_buffer.append(decoded_data)  # Salva i dati raw nel buffer parallelo (senza timestamp)

            if len(shift_buffer) == win_shift_samples:
                buffer.extend([d[0] for d in shift_buffer])  # Aggiungi solo i dati al buffer principale, ([d[0] for d in shift_buffer]) questo mi serve per prendere solo i dati senza timestamp
                shift_buffer.clear()  # Svuota il buffer di shift

                if len(buffer) == win_size_samples:
                    raw_batch = pd.DataFrame(list(raw_buffer), columns=['Channel 1', 'Channel 2', 'Channel 3'])  # Dati raw
                    data_frame = pd.DataFrame(buffer, columns=['Channel 1', 'Channel 2', 'Channel 3']) # Dati preprocessati, devo usare data_frame per coerenza con il codice di training
                    processed_data = preprocess_data(data_frame)
                    if processed_data is not None:
                        # Aggiungi dati preprocessati e raw alla coda con timestamp
                        data_queue.put((processed_data, raw_batch, time.time()))  # Timestamp attuale per calcolare il ritardo. viene calcolato il ritardo tra il momento in cui il batch viene processato e il momento in cui viene iniziata l'inferenza
                        processed_data_list.extend(processed_data.values.tolist())
        except Exception as e:
            print(f"Errore nel thread di acquisizione: {e}")
            break

# Thread di inferenza
def inference_thread():
    model = load_model(MODEL_CONFIG['path'])  # Carica il modello
    last_prediction_time = None  # Per calcolare la frequenza di predizione
    # Configura il dispositivo per l'inferenza
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Cuda disponibile" if torch.cuda.is_available() else "Cuda non disponibile")
    print(f"Modello spostato su {device}")
    max_queue_size = 2  # Numero massimo di batch da mantenere nella coda, mi serve quindi per gestire il ritardo tra acquisizione e inferenza
    
    while not stop_event.is_set():
        try:
            # Controlla se la coda ha più batch del consentito
            while data_queue.qsize() > max_queue_size:
                discarded_batch = data_queue.get()  # Scarta il batch più vecchio
                #print("Batch scartato per limitare il ritardo.")
            
            if not data_queue.empty():
                processed_data, raw_data_batch, batch_timestamp = data_queue.get()  # Ottieni il batch più recente
                current_time = time.time()

                # Calcola il ritardo tra acquisizione e inferenza
                delay = current_time - batch_timestamp
                print(f"Ritardo tra acquisizione e inferenza: {delay:.2f} s")

                # Calcola la frequenza di predizione
                if last_prediction_time is not None:
                    elapsed_time = current_time - last_prediction_time
                    if elapsed_time > 0:
                        prediction_frequency = 1 / elapsed_time
                        print(f"Frequenza di predizione: {prediction_frequency:.2f} Hz")
                        print(f"Tempo trascorso tra le predizioni: {elapsed_time:.2f} s")
                last_prediction_time = current_time

                # Effettua l'inferenza
                print("Dati raw:")
                print(raw_data_batch.head(5))  # Stampa un sottoinsieme dei dati raw
                processed_data =torch.tensor(processed_data.values).to(device)
                prediction = run_inference(model, processed_data)
                if prediction is not None:
                    print(f"Predizione per il batch: {prediction}")
                else:
                    print("Errore durante l'inferenza.")
        except Exception as e:
            print(f"Errore nel thread di inferenza: {e}")
            break

# Avvio dei thread
def main():
    acquisition = threading.Thread(target=acquisition_thread)
    inference = threading.Thread(target=inference_thread)
    acquisition.start()
    inference.start()

    try:
        while True:
            time.sleep(1)  # Mantieni il main thread attivo
    except KeyboardInterrupt:
        print("Interrotto dall'utente. Fermando i thread...")
        stop_event.set()
        acquisition.join()
        inference.join()

        # Salva i dati raw
        if raw_data_list:
            pd.DataFrame([d[0] for d in raw_data_list], columns=['Channel 1', 'Channel 2', 'Channel 3']).to_csv("raw_data.csv", index=False)
            print("Dati raw salvati in raw_data.csv")

        if processed_data_list:
            pd.DataFrame(processed_data_list, columns=['Channel 1', 'Channel 2', 'Channel 3']).to_csv("processed_data.csv", index=False)
            print("Dati preprocessati salvati in processed_data.csv")

    # Chiude la porta seriale
    finally:
        if ser.is_open:
            ser.close()
            print("Porta seriale chiusa.")

if __name__ == "__main__":
    main()
