from joblib import Parallel, delayed
import scipy as sp
import itertools
import gc
from tsfresh.feature_extraction import feature_calculators
import librosa
import numpy as np
import pywt # PyWavelets
import pandas as pd
from tqdm import tqdm

def cut_at_first_inf(TTF, acoustic_raw_data):
    """
    Taglia i vettori TTF e acoustic_raw_data fino al primo indice con 'inf' in TTF.
    
    Parametri:
        TTF (np.ndarray): Vettore di input con possibili valori 'inf'.
        acoustic_raw_data (np.ndarray): Vettore associato da tagliare allo stesso modo.
    
    Restituisce:
        tuple: TTF e acoustic_raw_data tagliati.
    """
    # Trova l'indice del primo elemento 'inf' in TTF
    first_inf_index = np.argmax(np.isinf(TTF))
    
    # Se non esiste un valore 'inf', restituisci i vettori interi
    if not np.isinf(TTF[first_inf_index]):
        print("Nessun valore 'inf' trovato in TTF.")
        return TTF, acoustic_raw_data
    
    # Stampa l'indice del primo 'inf'
    print("Indice del primo 'inf':", first_inf_index)
    
    # Taglia i vettori fino all'indice trovato
    TTF_cut = TTF[:first_inf_index]
    acoustic_raw_data_cut = acoustic_raw_data[:first_inf_index]
    
    return TTF_cut, acoustic_raw_data_cut

def denoise_signal_simple(x, wavelet='db4'):
    """
    Questa funzione riduce il rumore nel segnale  x  utilizzando wavelet denoising

    Input alla funzione:
	Il segnale  x  di input è una serie temporale di dimensione  n  (ad esempio, un array NumPy).
	
    Output della funzione:
	Il segnale ricostruito  x_denoised , ottenuto da pywt.waverec, ha in genere la stessa dimensione di  x .
    """
    coeff = pywt.wavedec(x, wavelet, mode="per")
    #univeral threshold
    uthresh = 10
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')
    # pywt.waverec: Ricostruisce il segnale a partire dai coefficienti wavelet modificati (dopo il thresholding).
	# Combina i coefficienti di approssimazione e dettaglio per ottenere il segnale denoisato.


def feature_gen(z, noise):
    """
    input: signal z (numpy array)
    z deve essere un segnale mono
    z deve essere un tensore di grado 1 (deve avere una dimensione)

    returns:
    A dataframe that contains 4 specific features of the signal z, in particular:

    1. number of picks of z
    2. 20 percentile di unna deviazione standard di una finestra mobile di dimensione 50 sul segnale  z 
    3-4. la media dei coefficienti MFCC 18 e 4 del segnale.
    """
    X = pd.DataFrame(index=[0], dtype=np.float64) # inizializza un dataframe con una sola riga vuota (senza colonne)
    
    z = z + noise              # add noise to the signal
    z = z - np.median(z)       # subtract median to the signal 

    den_sample_simple = denoise_signal_simple(z)  # denoise the signal

    mfcc = librosa.feature.mfcc(y=z)
    # estrae le caratteristiche MFCC (Mel-Frequency Cepstral Coefficients) da un segnale z
    # z deve essere un segnale mono
    # ritorna una matrice di dimensioni (n_mfcc, n_frames), dove
    # n_mfcc : Numero di coefficienti MFCC (di default, 20).
	# n_frames : Numero di finestre temporali in cui il segnale è stato suddiviso.
    # Il numero di finestre  n_frames dipende dalla durata della finestra (in campioni) e dal sovrapposizione (hop length)

    mfcc_mean = mfcc.mean(axis=1)
    # Calcola la media dei coefficienti MFCC lungo l’asse temporale (asse 1).
	# Ogni coefficiente MFCC descrive una particolare componente spettrale del segnale su tutta la durata.
    # Restituisce un array di dimensione (n_mfcc,)  (tipicamente  (20,) 

    rolled_std = pd.Series(z).rolling(50).std().dropna().values
    percentile_roll50_std_20 = np.percentile(rolled_std, 20)
    # pd.Series(z) trasforma z (numpy array) in un oggetto pandas.Series
    # Gli oggetti pandas.Series offrono metodi avanzati per l’elaborazione dei dati, come la rolling window.
    # .rolling(50) Applica una finestra mobile (rolling window) di lunghezza 50 alla serie.
    # Una rolling window divide la serie in segmenti consecutivi di lunghezza 50 (ad esempio, i primi 50 valori, poi i successivi 50 valori con un passo di 1, ecc.).
    # .std() Calcola la deviazione standard dei valori all’interno di ogni finestra mobile
    # .dropna() Rimuove i valori NaN (not-a-number) che compaiono all’inizio della serie rolling.
    # Infatti, Le prime finestre mobili non hanno abbastanza valori per calcolare la deviazione standard (ad esempio, per la prima finestra servono almeno 50 valori).
    # .values converte la serie pandas (con le deviazioni standard calcolate) in un array NumPy.
	# Questo perché: la funzione np.percentile lavora con array NumPy.
    # np.percentile(array, 20) Calcola il 20° percentile dell’array.
    # Il percentile indica il valore sotto il quale cade una certa percentuale dei dati. 
    # Ad esempio, il 20° percentile è il valore sotto il quale si trova il 20% dei valori.

    X['var_num_peaks_2_denoise_simple'] = feature_calculators.number_peaks(den_sample_simple, 2) 
    # conta il numero di picchi in un segnale x. Qui usiamo n = 2. I picchi sono quei punti i in cui:
    # 1. x[i] è maggiore di x[i-n] e x[i+n] e di tutti i valori intermedi nella finestra [i-n, i+n]
    # 2. Il valore in x[i] è almeno n più grande dei valori circostanti (nell'intervallo [i-n, i+n])
    # feature_calculators.number_peaks ritorna un intero

    X['var_percentile_roll50_std_20'] = percentile_roll50_std_20
    X['var_mfcc_mean18'] = mfcc_mean[18]
    X['var_mfcc_mean4'] = mfcc_mean[4]
    
    return X



def parse_sample(sample, start, chosen_noise):
    """
    sample è una parte del dataframe raw iniziale (contenente acustic_data e time_to_failure)
    la parte desiderata è uno spezzone che parte da uno degli indici dentro indices_to_calculate
    e finisce a indices_to_calculate + segment_size

    start è l'indice da cui parte la parte di dataframe che vogliamo


    ritorna un dataframe che contiene 
    1. le features del pezzo di segnale in sample['acoustic_data']
    2. un indice start del segnale
    3. l'ultimo valore di time to failure corrispondente al segnale in sample['acoustic_data'], cioè
        sarebbe il time to failure alla fine dell'acoustic signal di dimensione segment_size in questione
    """

    delta = feature_gen(sample['acoustic_data'].values, chosen_noise) # diventa un numpy array
    delta['start'] = start 
    # Aggiunge una colonna start a delta, contenente il valore di start                                           
    delta['target'] = sample['time_to_failure'].values[-1]
    # Aggiunge una colonna target a delta, contenente l’ultimo valore di time_to_failure del segmento.
    return delta    


    
def sample_train_gen(df, chosen_noise, segment_size=150_000, indices_to_calculate=[0]):
    """
    Ritorna un dataframe che contiene per ogni riga 6 colonne
    Ogni riga riguarda una partizione di dimensione segment_size del segnale acustico originale
    Il numero di righe quindi è numero di righe in df raw//segment_size (+1 nel caso in vui numero di righe di raw non sia multiplo di segment size)
    Le sei colonne sono:
    1. numero picchi di quella partizione del pezzo di segnale
    2. 20 percentile della rolling std di quel pezzo di segnale
    3. mfcc_mean18 di quel pezzo di segnale
    4. mfcc_mean4 di quel pezzo di segnale
    5. l'indice di start di quel segnale (l'indice in acoustic data nel dataframe raw)
    6. il time to failure alla fine di quel pezzo di segnale (ossia: quando finisce il pezzo
        di segnale in questione, quanto tempo manca al prossimo terremoto? (in secondi))
    """

    # eseguiamo la funzione parse_sample in modo parallelo su diversi segmenti di df
    # result è una lista di DataFrame ritornati da parse_sample per ogni segmento.
 
    result = Parallel(n_jobs=8, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(parse_sample)(df[int(i) : int(i) + segment_size], int(i), chosen_noise) 
                                                                                                for i in tqdm(indices_to_calculate))
    # result è una lista di df
    data = [r.values for r in result] # Estrae i valori dai DataFrame e diventano numpy array
    data = np.vstack(data)            #  impila verticalmente con np.vstack
    X = pd.DataFrame(data, columns=result[0].columns)
    X = X.sort_values("start")
    return X


def parse_sample_test(seg_id, noise):
    sample = pd.read_csv('/Users/andreagentilini/Desktop/Progetto_Chris/Coding_Chris/Data_Kaggle/test/' + seg_id + '.csv', dtype={'acoustic_data': np.int32})
    delta = feature_gen(sample['acoustic_data'].values, noise)
    delta['seg_id'] = seg_id
    return delta


def sample_test_gen(noise):
    X = pd.DataFrame()
    submission = pd.read_csv('/Users/andreagentilini/Desktop/Progetto_Chris/Coding_Chris/sample_submission.csv', index_col='seg_id')
    result = Parallel(n_jobs=4, temp_folder="/tmp", max_nbytes=None, backend="multiprocessing")(delayed(parse_sample_test)(seg_id, noise) for seg_id in tqdm(submission.index))
    data = [r.values for r in result]
    data = np.vstack(data)
    X = pd.DataFrame(data, columns=result[0].columns)
    return X


def generate_event_metadata(TTF_dense_no_inf_fixed, start_index):
    """
    Genera una lista di dizionari con 'start' e 'end' che rappresentano gli eventi.
    Ogni 'end' corrisponde a un indice in cui il valore è zero.
    'start' è l'indice immediatamente precedente a 'end', con il primo 'start' che parte da zero.
    
    Args:
        TTF_dense_no_inf_fixed (array-like): Vettore con valori, inclusi zeri.
    
    Returns:
        list: Lista di dizionari con chiavi 'start' e 'end'.
    """
    TTF_dense_no_inf_fixed = np.array(TTF_dense_no_inf_fixed)  # Garantisco che sia un array numpy
    etq_meta = []
    
    # Inizializza il primo start
    start = start_index
        
    
    # Trova tutti gli indici dove il valore è zero
    zero_indices = np.where(TTF_dense_no_inf_fixed == 0.0)[0]

    if int(start_index) != 0:
        for end in zero_indices:
            end = end + start_index
            if start < end:  # Evita duplicati
                etq_meta.append({"start": start, "end": end})
            start = end + 1  # Il prossimo start è l'attuale end
    
    for end in zero_indices:
        if start < end:  # Evita duplicati
            etq_meta.append({"start": start, "end": end})
        start = end + 1  # Il prossimo start è l'attuale end
    
    return etq_meta


def find_train_size(TTF_dense_no_inf_fixed, fraction=0.8):
    """
    Trova l'indice train_size per dividere il dataset in terremoti completi.
    Un terremoto completo è indicato da valori zero in TTF_dense_no_inf_fixed.

    Args:
        TTF_dense_no_inf_fixed (array-like): Array o lista con i valori TTF.
        fraction (float): Percentuale desiderata per il train set (default = 0.8).

    Returns:
        int: Indice train_size che rappresenta la fine di un terremoto completo.
    """
    # Converti in array numpy per facilità
    TTF_dense_no_inf_fixed = np.array(TTF_dense_no_inf_fixed)
    
    # Trova tutti gli indici dove TTF è zero
    zero_indices = np.where(TTF_dense_no_inf_fixed == 0.0)[0]
    
    # Calcola l'indice target come l'80% della lunghezza totale
    target_index = int(fraction * len(TTF_dense_no_inf_fixed))
    
    # Trova l'indice zero più vicino ma non oltre il target_index
    train_size = max([i for i in zero_indices if i <= target_index])
    
    return train_size