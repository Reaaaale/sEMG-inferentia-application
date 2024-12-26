from scipy.signal import butter, filtfilt
import numpy as np
import struct
from config.config import FILTER_CONFIG, TRAINING_PERCENTILES, SPIKE_CONFIG
import torch



def decode_data(data):
    """
    Decodifica i dati grezzi ricevuti dalla porta seriale in 3 canali.

    Args:
        data: Byte array contenente i dati.

    Returns:
        Una tupla con i valori dei tre canali o None in caso di errore.
    """
    if len(data) != 4:
        print(f"Errore: Lunghezza dati non valida ({len(data)})")
        return None

    try:
        data = data[::-1]  # Reverse the byte order
        sample_data = struct.unpack('>l', data)[0]

        # Extract values for 3 channels
        channel1 = (sample_data >> 20) & 0x3FF
        channel2 = (sample_data >> 10) & 0x3FF
        channel3 = sample_data & 0x3FF

        #print(f"Decoded: Channel1={channel1}, Channel2={channel2}, Channel3={channel3}")
        return channel1, channel2, channel3
    except Exception as e:
        print(f"Errore nella decodifica: {e}")
        return None
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def butter_highpass_filter(data, cutoff_high, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_high / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()
def normalize_data(data_frame, training_percentiles):
    
    for channel in ['Channel 1', 'Channel 2', 'Channel 3']:
        min_val = training_percentiles[channel]['min']
        max_val = training_percentiles[channel]['max']
        data_frame[channel] = np.clip((data_frame[channel] - min_val) / (max_val - min_val), 0, 1)
    return data_frame


def process_data(array, max_derivative_order, delta_value, ch_start=None, ch_end=None):
    """
    Processes an array of data by expanding it based on derivative order and generating a spike array.
    """
    array = array[:, ch_start:ch_end]
    old_dim_size = array.shape[1]
    new_dim_size = old_dim_size * (max_derivative_order + 1)

    spike_array = np.zeros((array.shape[0] - max_derivative_order * 4, new_dim_size * 2), dtype=np.int8)
    expanded_array = np.zeros((array.shape[0], new_dim_size))
    expanded_array[:, :old_dim_size] = array

    for n in range(1, max_derivative_order + 1):
        for i in range(old_dim_size):
            for j in range(array[:, i].shape[0] - n * 4):
                expanded_array[j + n * 2, old_dim_size * n + i] = (
                    - expanded_array[j, old_dim_size * (n - 1) + i]
                    - 2 * expanded_array[j + 1, old_dim_size * (n - 1) + i]
                    + 2 * expanded_array[j + 2, old_dim_size * (n - 1) + i]
                    + expanded_array[j + 3, old_dim_size * (n - 1) + i]
                )

    expanded_array = expanded_array[max_derivative_order * 2 : - max_derivative_order * 2, :]

    for n in range(max_derivative_order + 1):
        for i in range(old_dim_size):
            dc_val = expanded_array[0, old_dim_size * n + i]
            for k, j in enumerate(expanded_array[:, old_dim_size * n + i]):
                if j > dc_val + delta_value[n]:
                    dc_val = j
                    spike_array[k, (old_dim_size * n + i) * 2] = 1
                    spike_array[k, (old_dim_size * n + i) * 2 + 1] = 0
                elif j < dc_val - delta_value[n]:
                    dc_val = j
                    spike_array[k, (old_dim_size * n + i) * 2] = 0
                    spike_array[k, (old_dim_size * n + i) * 2 + 1] = 1
                else:
                    spike_array[k, (old_dim_size * n + i) * 2] = 0
                    spike_array[k, (old_dim_size * n + i) * 2 + 1] = 0

    return expanded_array, spike_array

def load_model(model_path):
    try:
        # Determina il dispositivo (CUDA o CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Carica il modello e spostalo sul dispositivo corretto
        model = torch.load(model_path, map_location=device)
        model.to(device)  # Sposta il modello sul dispositivo
        model.eval()  # Imposta il modello in modalità di valutazione
        print(f"Modello caricato con successo su {device}!")
        return model
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        exit()

        
def preprocess_data(data_frame):
    try:
        # Filtraggio
        for channel in ['Channel 1', 'Channel 2', 'Channel 3']:
            data_frame[channel] = butter_lowpass_filter(data_frame[channel], FILTER_CONFIG['cutoff_lowpass'], FILTER_CONFIG['fs'], FILTER_CONFIG['order']) #passa-basso
            data_frame[channel] = butter_highpass_filter(data_frame[channel], FILTER_CONFIG['cutoff_highpass'], FILTER_CONFIG['fs'], FILTER_CONFIG['order']) #passa-alto
            data_frame[channel] = moving_average(data_frame[channel], FILTER_CONFIG['window_size']) #media-mobile
        
        # Normalizzazione
        data_frame = normalize_data(data_frame, TRAINING_PERCENTILES)

        return data_frame
    except Exception as e:
        print(f"Errore durante il preprocessamento: {e}")
        return None

def run_inference(model, processed_data):
    try:
        # Genera spike array, richiamo la funzione process_data utilizzata in fase di training
        emg_array = processed_data[['Channel 1', 'Channel 2', 'Channel 3']].to_numpy()
        _, spike_array = process_data(
            emg_array,
            SPIKE_CONFIG['max_derivative_order'],
            SPIKE_CONFIG['delta_values']
        )

        # Preparo l'input per il modello replicando quello del training
        spike_array = spike_array.T  # Trasposizione dell'array
        input_tensor = torch.tensor(spike_array, dtype=torch.float32).unsqueeze(0)  # [1, num_channels, sequence_length]

        # Print per debug
        print(f"Input tensor shape before model: {input_tensor.shape}")

        # Esegui inferenza
        output = model(input_tensor)

        #Debug dell'output
        #print(f"Output tensor shape: {output.shape}, values: {output}")

        # Riduzione lungo la dimensione della sequenza (dim=2)
        reduced_output = output.sum(dim=2)  # Ora il tensore è [batch_size, num_classes]

        # Estrai la predizione, estrae la classe più volte predetta all'interno della finestra. 
        prediction = torch.argmax(reduced_output, dim=1).item()

        return prediction
    except Exception as e:
        print(f"Errore durante l'inferenza: {e}")
        return None
