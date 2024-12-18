import numpy as np
import os
import matplotlib.pyplot as plt
import re


def create_data_chunk(input_folder,list_channels,chunk,float_type=None):
    '''
    Purpose
    ---------
        To create an array representing a specific chunk of experimental data from multiple channels stored in binary files.

    Parameters
    ---------
       input_folder: folder of experiment
       list_channels: list channels to consider (e.g. [1,2,5], consider channels 1,2,5)
       chunk: chunk number
       float_type: or 'float16' o 'float32' or 'float64'

    Returns
    ---------
       data_chunk: array of the chunk with shape=(number_ch,1/fs*len(1chunck))

    '''
    number_ch=len(list_channels)
    data_chunk=[[] for i in range(number_ch)]
    chunk_name=input_folder+'ae_chunk'+str(chunk)+'/'

    if not os.path.exists(chunk_name) or not os.path.isdir(chunk_name):
        raise FileNotFoundError(f"The folder '{chunk_name}' does not exist.")

    for i in range(len(data_chunk)):
        ch_n=int(list_channels[i])
        fileName=chunk_name+'ae_chunk'+str(chunk)+'_ch'+str(ch_n)+'.bin'

        if not os.path.exists(fileName) or not os.path.isfile(fileName):
            raise FileNotFoundError(f"The file '{fileName}' does not exist.")

        # Data can be saved in different format
        with open(fileName, mode='rb') as file: # b is important -> binary
            if float_type=='float32':
                content=np.frombuffer(file.read(),dtype='float32')
                content=np.array(content[32::].tolist())
            elif float_type=='float16':
                content=np.frombuffer(file.read(),dtype='float16')
                content=np.array(content[64::].tolist())
            elif float_type=='float64':
                content=np.frombuffer(file.read(),dtype='float64')
                content=np.array(content[16:].tolist())

            else:
                raise ValueError("float type is not valid! Put float16, float32 or float64")
        data_chunk[i]=content

    return np.array(data_chunk)


def plot_chunk(fs,data_chunk,channels_names,chunkname=None,save_flag=False,save_cd=None,name_file=None):

    """
    Purpose
    ---------
        Plot Time-series of more channels and chunks

    Parameters
    ---------
       fs: sampling rate
       data_chunk: chunk you want to plot
       channels_names: list of the channels
       chunkname: name of the chunk
       save_flag: bolean, if True, save the plot
       save_cd: folder where you save the plots
       name_file: figure name
       list_peaks: if not None, plot all the lpeaks for each channel
       list_peaks_akaike: if not None, plot all the akaike for each channel

    Returns
    ---------
       Plot all the channels and eventual picks

    """
    T=np.array(range(0,data_chunk.shape[1]))*(1/fs)
    number_ch=len(channels_names)
    fig, axs = plt.subplots(number_ch, 1,figsize=(10,10),sharex=True)
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=.0)

    for i in range(len(axs)):
        col=plt.cm.viridis(i / number_ch)
        axs[i].plot(T,data_chunk[i,:], linewidth=0.5,color=col)
        axs[i].text(.5,.92,'Channel: '+str(channels_names[i]),horizontalalignment='center',
                 transform=axs[i].transAxes,fontsize=7,color=col,bbox=dict(facecolor='white', edgecolor='k',boxstyle='round'))
        axs[i].yaxis.set_tick_params(labelsize=5)
        axs[i].xaxis.set_tick_params(labelsize=5)
        axs[i].set_ylabel(r'$Voltage\hspace{0.4}[V]$',fontsize=5)
        axs[i].set_xlim([T[0],T[-1]])

    axs[i].set_xlabel('Time [s]')

    if chunkname is not None:
        axs[0].set_title('Chunk: '+str(chunkname))

    if save_flag==True:
        if name_file is None:
            raise ValueError("name_file should not be None to prevent accidental file saving.")
        else:
            fileName=save_cd+name_file+'.png'
        if not os.path.exists(fileName) or not os.path.isfile(fileName):
            plt.savefig(fileName)
            plt.close()
        else:
            raise FileNotFoundError(f"The file '{fileName}' already exists.")
    return



def create_data_chunk_single_channel(input_folder, channel, chunk, float_type=None):
    '''
    Purpose
    ---------
    This function creates an array representing a chunk of signal acquired from a single channel stored in a binary file.

    Parameters
    ---------
    input_folder: experiment folder
    channel: channel number to consider (e.g., 1, 2, 5)
    chunk: chunk number
    float_type: 'float16', 'float32', or 'float64'

    Returns
    ---------
    data_chunk: array containing the specified channel's data for the chunk
    '''
    # Create the relative path to the chunk folder
    chunk_name = os.path.join(input_folder, 'ae_chunk' + str(chunk))
    
    if not os.path.exists(chunk_name) or not os.path.isdir(chunk_name):
        raise FileNotFoundError(f"The folder '{chunk_name}' does not exist.")

    # Build the filename corresponding to the requested channel
    fileName = os.path.join(chunk_name, f'ae_chunk{chunk}_ch{channel}.bin')

    if not os.path.exists(fileName) or not os.path.isfile(fileName):
        raise FileNotFoundError(f"The file '{fileName}' does not exist.")
    
    with open(fileName, mode='rb') as file:
        if float_type == 'float32':
            content = np.frombuffer(file.read(), dtype='float32')
            # Remove potential header (offset 32)
            content = np.array(content[32:].tolist())
        elif float_type == 'float16':
            content = np.frombuffer(file.read(), dtype='float16')
            # Remove potential header (offset 64)
            content = np.array(content[64:].tolist())
        elif float_type == 'float64':
            content = np.frombuffer(file.read(), dtype='float64')
            # Remove potential header (offset 16)
            content = np.array(content[16:].tolist())
        else:
            raise ValueError("Invalid float_type! Use 'float16', 'float32', or 'float64'.")

    return content


def combine_chunks_single_channel(folder_path, chunk, channel, step, T, float_type=None):
    '''
    Purpose
    ---------
        Combina una serie di chunk (segmenti temporali di dati sperimentali) in un unico array continuo,
        considerando un singolo canale.

    Parameters
    ---------
       folder_path: str
           Cartella dell'esperimento.
       chunk: int
           Numero del primo chunk da considerare.
       channel: int
           Il singolo canale da cui estrarre i dati.
       step: int
           Numero di chunk da concatenare (dal chunk iniziale fino a chunk+step-1).
       T: array-like
           Base temporale di un singolo chunk (usata per determinare la lunghezza dei dati di ciascun chunk).
       float_type: str, opzionale
           Tipo di floating point ('float16' o 'float32'), altrimenti default 'float64'.

    Returns
    ---------
       data_chunk: np.ndarray (1D)
           Array contenente i dati concatenati di tutti i chunk per il canale specificato.
       chunk_numbers: list
           Lista dei numeri di chunk considerati.
    '''

    chunk_numbers = list(range(int(chunk), int(chunk) + step))
    num_iterations = len(chunk_numbers)
    chunk_length = len(T)

    # Inizializza l'array finale
    data_chunk = np.zeros(chunk_length * num_iterations, dtype=float_type if float_type else float)

    # Itera sui chunk e concatena i dati uno di seguito all'altro
    for i, c in enumerate(chunk_numbers):
        start_idx = i * chunk_length
        end_idx = start_idx + chunk_length
        data_chunk[start_idx:end_idx] = create_data_chunk_single_channel(folder_path, channel, c, float_type=float_type)

    return data_chunk, chunk_numbers


def decimate_f(array,fs,target_sampling_rate):

    '''
    Purpose
    ---------
        Undersampling an array with scipy.signal.decimate

    Parameters
    ---------
       array: input folder path
       fs: input sampling rate
       target_sampling_rate: target for the resampled array
    Returns
    ---------
       array_decimated: array with the new sampling rate

    '''
    from scipy.signal import decimate
    decimation_factor = int(fs / target_sampling_rate)
    return decimate(array, q=decimation_factor, zero_phase=True)


def combine_chunks(folder_path, chunk, list_channels, step, T,float_type=None):

    '''
    Purpose
    ---------
        To combine data from a series of chunks (time-segmented experimental data files) into one continuous array, while maintaining the sequence of data across all specified channels.

    Parameters
    ---------
       folder_path: folder of experiment
       chunk: first chunk number
       list_channels: list channels to consider (e.g. [1,2,5], consider channels 1,2,5)
       step: consider N chunks from chunk to chunk+step
       T: length in Time of the first chunk
       float_type: or 'float16' o 'float32'

    Returns
    ---------
       data_chunk: combined data_chunk
       chunk_numbers: list of chunk numbers

    '''
    chunk_numbers = list(range(int(chunk), int(chunk) + step))
    num_iterations = len(chunk_numbers)
    number_ch = len(list_channels)

    data_chunk = np.zeros((number_ch, len(T) * num_iterations))

    # Simulating the appending process in a loop
    array_size = (number_ch, len(T))  # Size of the array to be appended in each iteration

    for i, chunk in enumerate(chunk_numbers):
        # Determine the indices to insert the new array

        start_idx = i * array_size[1]
        end_idx = start_idx + array_size[1]
        # Fill the final array with the new array at the appropriate indices
        data_chunk[:, start_idx:end_idx] = create_data_chunk(folder_path, list_channels, chunk,float_type=float_type)

    return data_chunk


def reduce_stress_from_acoustic(folder_path,starting_chunk,fs,fs_red,step=20,f_chunk=None,float_type=None,last_numb=None,first_numb=None):
    '''
    Purpose
    ---------
        To process and downsample shear stress data from acoustic data to a target sampling rate

    Parameters
    ---------
       folder_path: folder where all the chunk are saved
       fs: real sampling rate
       fs_red: target sampling rate
       step: consider N chunks from chunk to chunk+step
       f_chunk: number of points of each chunk
                (fake frequency put int the Tipie acquisition file, which is not the real frequency)
       float_type: or 'float16' o 'float32'
       last_numb: if is an integer, then reduce stress until the chunk number last_numb
       first_numb: if is an integer, then reduce stress from the chunk number first_numb

    Returns
    ---------
       Se_arrayT: the stress decimated at the target sampling rate fs_red

    '''

    folder_names=list_folders(folder_path)

    # Extract numberic folder numbers
    if last_numb is not None:
        folder_numbers=last_numb
    else:
        folder_numbers = [extract_number(folder) for folder in folder_names]
        folder_numbers = [num for num in folder_numbers if num is not None]  # Filtra i None

        if folder_numbers:  # Se ci sono numeri validi
            max_number = max(folder_numbers)
            print(f"Max folder number: {max_number}")
        else:
            print("No valid folder numbers found!")
            raise ValueError("No valid folder numbers found. Check folder names or folder path.")
            
    if first_numb is not None:
        chunk=first_numb
    else:
        chunk=starting_chunk  

    Se_arrayT=[]
    print('The last chunk is the number: ', max_number)

    # Loop through chunks and process data
    for i in range(chunk,max_number,step):
        print(f"Processing chunk: {i}")
        Se_array=combine_chunks(folder_path,chunk=i,list_channels=[1],step=step,T=np.arange(f_chunk),float_type=float_type)
        Se_arrayT.append(decimate_f(Se_array.flatten(),fs,fs_red))

    return np.concatenate(np.array(Se_arrayT))


def list_folders(main_folder):

    '''
    Purpose
    ---------
        The function list_folders takes a directory path (main_folder) as input and returns a list of all the subfolders (directories) within that path.
    '''
    folders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]
    return folders

def extract_number(folder_name):

    '''
    Purpose
    ---------
        The extract_number function extracts numeric parts from a given folder (or string) name using a regular expression.
    '''
    # Use regular expression to extract the numeric part from the folder name
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else None


def unify_close_indices(significant_points, threshold=5):
    """
    Unifies consecutive indices in an array if the difference between them is less than 'threshold'.
    Only the first element of each group is retained.

    Parameters:
        significant_points (list or array): Array of indices to be unified.
        threshold (int): Maximum allowed distance to unify consecutive indices.

    Returns:
        list: A list of unified indices, retaining only the first of each group.
    """
    # Convert input to a NumPy array if it's not already
    significant_points = np.asarray(significant_points)

    if significant_points.size == 0:  # Check for empty array
        return []

    unified_indices = [int(significant_points[0])]  # Start with the first index

    for i in range(1, len(significant_points)):
        if significant_points[i] - significant_points[i-1] > threshold:
            unified_indices.append(int(significant_points[i]))  # Append only if far enough

    return np.array(unified_indices)

def convert_to_arbitrary_units(data, v_min, v_max, n_bits=16):
    """
    Convert a float32 array to arbitrary units (integer scale).
    
    Parameters:
    - data (ndarray): Input float32 array.
    - v_min (float): Minimum voltage value in the input range.
    - v_max (float): Maximum voltage value in the input range.
    - n_bits (int): Bit depth for output (default is 16-bit).

    Returns:
    - ndarray: Integer array in arbitrary units.
    """
    # Ensure the input is float32
    data = data.astype(np.float32)
    
    # Define the maximum value based on bit depth
    max_int_value = (2 ** n_bits) - 1
    
    # Scale data linearly to the range [0, max_int_value]
    scaled_data = (data - v_min) / (v_max - v_min) * max_int_value
    
    # Clip the data to ensure it stays in range [0, max_int_value]
    scaled_data_clipped = np.clip(scaled_data, 0, max_int_value)
    
    # Convert to integers
    return scaled_data_clipped.astype(np.int32)

