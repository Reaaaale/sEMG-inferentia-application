# Configurazione dei percentili calcolati sui dati di training
TRAINING_PERCENTILES = {
    'Channel 1': {'min': -38.569215446019314, 'max': 37.0587119464698},
    'Channel 2': {'min': -16.451727023218243, 'max': 2.9519059825504774},
    'Channel 3': {'min': -3.0587448762961658, 'max': 9.511034990176089}
}

# Configurazione dei parametri di filtraggio
FILTER_CONFIG = {
    'fs': 1000,  # Frequenza di campionamento in Hz
    'cutoff_lowpass': 400,  # Frequenza di taglio in Hz per il filtro passa-basso
    'cutoff_highpass': 20,  # Frequenza di taglio in Hz per il filtro passa-alto
    'order': 5,  # Ordine del filtro Butterworth
    'window_size': 100,  # Numero di campioni per la finestra di media mobile
    'buffer_size': 16  # Numero minimo di campioni richiesti nel buffer
}

# Configurazione della generazione spike
SPIKE_CONFIG = {
    'max_derivative_order': 2,
    'delta_values': [0.015, 0.015, 0.015],  # Devono avere la stessa lunghezza dei canali
}

# Configurazione delle finestre per la pipeline
WINDOW_CONFIG = {
    'win_size': 0.5,       # Durata della finestra in secondi
    'win_shift': 0.01,     # Spostamento tra finestre consecutive in secondi
}

# Configurazione della porta seriale
SERIAL_CONFIG = {
    'port': 'COM4',  # Porta seriale
    'baud_rate': 500000  # Velocità della porta seriale
}

# Configurazione modello
MODEL_CONFIG = {
    'path': r"C:\Users\alucd\Downloads\model_best.pt"
}
