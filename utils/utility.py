import numpy as np
import pandas as pd
import os, re, math, h5py, random, itertools, platform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from scipy.signal import spectrogram, welch, stft
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, auc, average_precision_score, roc_curve, confusion_matrix
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, TensorBoard
from matplotlib.colors import LinearSegmentedColormap
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DynamicThresholdCallback(Callback):
    def __init__(self):
        super(DynamicThresholdCallback, self).__init__()
        self.validation_losses = []
        
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        self.validation_losses.append(val_loss)
        
    def get_dynamic_threshold(self):
        return np.mean(self.validation_losses) + np.std(self.validation_losses)

def check_os():
    # Get the OS information and convert to lowercase
    return platform.system().lower()

def clear_screen(os_type):
    # Clear the console screen based on the OS type
    command = 'cls' if os_type == 'windows' else 'clear'
    os.system(command)

def process_files_test(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.endswith('.csv') or filename.endswith('.log'):
                # Build full file path
                full_path = os.path.join(dirpath, filename)
                
                # Debug statement for the filename
                print(f"Processing: {filename}")
                
                # Read the file into a DataFrame
                if filename.endswith('.csv'):
                    df = pd.read_csv(full_path)
                    # Rename columns
                    df.rename(columns={'status': 'Sts', 'x': 'x_axis', 'y': 'y_axis', 'z': 'z_axis', 'time': 'datetime'}, inplace=True)
                else:  # For log files
                    data_list = []
                    with open(full_path, 'r') as file:
                        for line in file.readlines():
                            items = line.strip().split(",")
                            values = [item.split(":")[1] if ":" in item else item for item in items[:-1]]
                            datetime_value = items[-1]
                            values.append(datetime_value)
                            data_list.append(values)
                    df = pd.DataFrame(data_list, columns=['Sts', 'x_axis', 'y_axis', 'z_axis', 'datetime'])
                
                try:
                    # Convert to datetime
                    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')                    
                    
                    # Additional columns for date and time
                    df['date'] = df['datetime'].dt.date
                    df['time'] = df['datetime'].dt.time
                    
                    # Remove 'stage' NULL column if detected.
                    if 'stage' in df.columns:
                        df.drop(columns=['stage'], inplace=True)

                except KeyError as ke:
                    print(f"KeyError: {ke}")
                except ValueError as ve:
                    print(f"ValueError: {ve}")

                # Save DataFrame to h5 format in the original folder
                h5_path = os.path.join(dirpath, filename.split('.')[0] + '.h5')
                # print(df['datetime'].head()) # Debugging
                df.to_hdf(h5_path, key='df', mode='w')
                
                # Debug statement for saving
                print(f"Saved: {filename} to {h5_path}")

def load_h5_files_test2(root_path):
    machine_data = []  # Simplified for example
    total_h5_file_counter = 0 # Keep track of .h5 files found

    is_first_iteration = True # to capture machine_type and persist
    machine_type = None # variable to store the machine_type

    for dirpath, dirnames, filenames in os.walk(root_path):        
        # Extract folder name from the directory path
        folder_name = os.path.basename(dirpath)
        # Debug print to track where we are
        print(f"Current folder: {folder_name}")

        if is_first_iteration:  # <-- New condition
            machine_type = folder_name
            is_first_iteration = False  # <-- Reset the flag

        absolute_dirpath = os.path.abspath(dirpath)
        print(f"Currently in directory (Absolute Path): {absolute_dirpath}")  # Debugging statement

        parts = absolute_dirpath.split(os.sep)

        # Identify status dynamically
        status = None
        for part in reversed(parts):
            if part.lower() in ['normal', 'abnormal']:
                status = part.lower()

        print(f"Status after identification: '{status}', Machine Type: '{machine_type}'")  # Debugging statement

        if status:
            h5_files_count = 0
            # print(f"Status is {status}. Starting to count .h5 files.")  # Debugging statement

            for filename in filenames:
                if filename.endswith('.h5'):
                    # print(f"Found .h5 file: {filename}")  # Debugging statement
                    h5_files_count += 1
                    total_h5_file_counter += 1

                    try:
                        full_h5_path = os.path.join(dirpath, filename)
                        df = pd.read_hdf(full_h5_path, key='df')  # Assuming 'df' is the key used while saving
                        machine_data.append(df)
                        # print('df appended....')
                        # column_names = df.columns.tolist()
                        # print(f"Column names in {filename}: {column_names}")
                        # print(df.head()) # Debugging
                    except Exception as e:
                        print(f"An error occurred while reading {filename}: {e}")

            print(f"Number of .h5 files in this folder: {h5_files_count}")  # Debugging statement
    
    print(f"Total number of .h5 files identified: {total_h5_file_counter}")  # Print total h5 files found thus far.

    return machine_data 

def preprocess_data_with_padding(df_dict, max_length=None):
    preprocessed_data = []
    scaler = StandardScaler()

    # Find the length of the longest sequence if max_length is not provided
    if max_length is None:
        max_length = max(len(df) for df in df_dict.values())

    for filename, df in df_dict.items():
        # Padding
        pad_length = max_length - len(df)
        if pad_length > 0:
            padded_data = np.pad(df, [(0, pad_length), (0, 0)], mode='constant')
        else:
            padded_data = df.values  # Convert DataFrame to numpy array
        
        # Feature scaling
        scaled_data = scaler.fit_transform(padded_data)
        
        preprocessed_data.append(scaled_data)

    return preprocessed_data, max_length

# Takes a list of time-series arrays and returns a list of feature-engineering arrays.
# Note that the input arrays are interpreted in the following order ['Sts', 'x_axis', 'y_axis', 'z_axis'], row index is datetime
def feature_engineering(preprocessed_data):
    engineered_features_list = []

    for data in preprocessed_data:
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        skewness = skew(data, axis=0)
        kurt = kurtosis(data, axis=0)

        # Concatenate all the features to form a single feature vector
        feature_vector = np.hstack([mean, std_dev, skewness, kurt])

        # Output list is read as ['Sts', 'x_axis', 'y_axis', 'z_axis', 'mean, 'std_dev', 'skewness', kurtosis]
        engineered_features_list.append(feature_vector)
    
    return engineered_features_list

def prepare_train_test2(preprocessed_normal, preprocessed_abnormal, en_feature_eng=False, split_ratio=0.8):

    # To flatten 3D array to 2D array when feature_engineering is not enabled.
    # E.g., preprocessed_normal size is (300, 298230, 4); preprocessed_abnormal size is (60, 298230, 4).
    def flatten_3D_to_2D(data):
        # Flatten each 3D array to a 2D array by reshaping it
        return [array.flatten() for array in data]

    # Check if the arrays are 3D
    if len(preprocessed_normal[0].shape) == 3:
        preprocessed_normal = flatten_3D_to_2D(preprocessed_normal)
    if len(preprocessed_abnormal[0].shape) == 3:
        preprocessed_abnormal = flatten_3D_to_2D(preprocessed_abnormal)

    # Initialize empty lists to store the concatenated arrays
    X_train, y_train, X_test, y_test = [], [], [], []
    
    # Helper function to add instances to datasets
    def add_instances(X_list, y_list, data_list, label):
        for array in data_list:
            X_list.append(array)
            y_list.append(label)
            # y_list.append(np.full(array.shape[0], label))  # Fill a numpy array with the given label

    # Calculate the number of abnormal and normal instances to include in the training set
    num_abnormal_train = int(len(preprocessed_abnormal) * split_ratio)
    num_normal_train = int(len(preprocessed_normal) * split_ratio)

    # Add normal instances to both training and test sets
    add_instances(X_train, y_train, preprocessed_normal[:num_normal_train], 0)
    add_instances(X_test, y_test, preprocessed_normal[num_normal_train:], 0)
    add_instances(X_train, y_train, preprocessed_abnormal[:num_abnormal_train], 1)
    add_instances(X_test, y_test, preprocessed_abnormal[num_abnormal_train:], 1)

    #Return numpy array
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if not en_feature_eng:
        y_train_repeated, y_test_repeated = [], []

        for label, dataset, target_list in zip([0, 1], [preprocessed_normal, preprocessed_abnormal], [y_train_repeated, y_test_repeated]):
            num_train = int(len(dataset) * split_ratio)
            for i, segment in enumerate(dataset):
                repeat_times = len(segment)
                if i < num_train:
                    y_train_repeated.extend([label] * repeat_times)
                else:
                    y_test_repeated.extend([label] * repeat_times)

        y_train = np.array(y_train_repeated)
        y_test = np.array(y_test_repeated)
        
        # If you want to reshape X_train and X_test here, you can also do that
        if len(X_train.shape) > 1:
            X_train = X_train.reshape(-1, X_train.shape[-1])
        if len(X_test.shape) > 1:
            X_test = X_test.reshape(-1, X_test.shape[-1])

    return X_train, y_train, X_test, y_test

# For LSTM train-test split data prepation only.
def prepare_train_test3(preprocessed_normal, preprocessed_abnormal, en_feature_eng=False, split_ratio=0.8):
    def flatten_3D_to_2D(data):
        return data.reshape(data.shape[0], -1)

    if not en_feature_eng:
        # Ensure that all elements have the same shape in 3D before flattening.
        assert all(elem.shape == preprocessed_normal[0].shape for elem in preprocessed_normal), "Normal data has inconsistent shapes."
        assert all(elem.shape == preprocessed_abnormal[0].shape for elem in preprocessed_abnormal), "Abnormal data has inconsistent shapes."

        preprocessed_normal = flatten_3D_to_2D(np.array(preprocessed_normal))
        preprocessed_abnormal = flatten_3D_to_2D(np.array(preprocessed_abnormal))

    num_abnormal_train = int(len(preprocessed_abnormal) * split_ratio)
    num_normal_train = int(len(preprocessed_normal) * split_ratio)
    print(f'Training set - Number of Abnormal data are f{num_abnormal_train}')
    print(f'Training set - Number of Normal data are f{num_normal_train}')

    # Flatten all arrays and append to train/test lists.
    X_train = np.vstack((preprocessed_normal[:num_normal_train], preprocessed_abnormal[:num_abnormal_train]))
    X_test = np.vstack((preprocessed_normal[num_normal_train:], preprocessed_abnormal[num_abnormal_train:]))
    
    # Create labels.
    y_train = np.array([0] * num_normal_train + [1] * num_abnormal_train)
    y_test = np.array([0] * (len(preprocessed_normal) - num_normal_train) + [1] * (len(preprocessed_abnormal) - num_abnormal_train))
    
    return X_train, y_train, X_test, y_test



# For testing only - may be deprecated in future.
def prepare_train_test_1(preprocessed_normal, preprocessed_abnormal, algos_to_run, split_ratio=0.8):

    def flatten_3D_to_2D(data):
        # Flatten each 3D array to a 2D array by reshaping it
        return [array.flatten() for array in data]

    # Checking if 'LSTM-Autoencoder' is in the algos_to_run
    lstm_autoencoder_present = 'LSTM-Autoencoder' in algos_to_run
    
    # Flatten the data if LSTM-Autoencoder is not present
    if not lstm_autoencoder_present:
        preprocessed_normal = flatten_3D_to_2D(preprocessed_normal)
        preprocessed_abnormal = flatten_3D_to_2D(preprocessed_abnormal)

    # Calculate the number of samples for training and testing based on the split_ratio
    normal_split_index = int(len(preprocessed_normal) * split_ratio)
    abnormal_split_index = int(len(preprocessed_abnormal) * split_ratio)

    # Create the train and test sets
    X_train = np.array(preprocessed_normal[:normal_split_index] + preprocessed_abnormal[:abnormal_split_index])
    y_train = np.array([0] * normal_split_index + [1] * abnormal_split_index)
    X_test = np.array(preprocessed_normal[normal_split_index:] + preprocessed_abnormal[abnormal_split_index:])
    y_test = np.array([0] * (len(preprocessed_normal) - normal_split_index) + [1] * (len(preprocessed_abnormal) - abnormal_split_index))

    # If the data is for LSTM Autoencoder, it should be padded to the same length
    if lstm_autoencoder_present:
        # Pad sequences to the same length
        max_length = max(max(map(len, preprocessed_normal)), max(map(len, preprocessed_abnormal)))
        X_train = pad_sequences(X_train, maxlen=max_length, padding='post', dtype='float32')
        X_test = pad_sequences(X_test, maxlen=max_length, padding='post', dtype='float32')

    # Debugging output
    print(f'normal_split_index: {normal_split_index}, abnormal_split_index: {abnormal_split_index}')
    print(f'Shape of preprocessed_normal: {np.shape(preprocessed_normal)}')
    print(f'Shape of preprocessed_abnormal: {np.shape(preprocessed_abnormal)}')
    print(f'Shape of X_train: {np.shape(X_train)}')
    print(f'Shape of y_train: {np.shape(y_train)}')
    print(f'Shape of X_test: {np.shape(X_test)}')
    print(f'Shape of y_test: {np.shape(y_test)}')     

    return X_train, y_train, X_test, y_test


def plot_raw_vibration(data, status):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    axes = axes.flatten()

    axes[0].plot(data['x_axis'])
    axes[0].set_title(f'X-Axis Vibration - {status}')

    axes[1].plot(data['y_axis'])
    axes[1].set_title(f'Y-Axis Vibration - {status}')

    axes[2].plot(data['z_axis'])
    axes[2].set_title(f'Z-Axis Vibration - {status}')

    plt.tight_layout()
    plt.show()

# New function for side-by-side comparison
def plot_comparison(final_df, selected_datetime=None):
    if selected_datetime:
        final_df = final_df[final_df['datetime'] == selected_datetime]
        
    machine_ids = final_df['machine_id'].unique()
    
    for machine_id in machine_ids:
        df_machine = final_df[final_df['machine_id'] == machine_id]
        df_normal = df_machine[df_machine['status'] == 'normal']
        df_abnormal = df_machine[df_machine['status'] == 'abnormal']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        
        axes[0][0].plot(df_normal['x_axis'])
        axes[0][0].set_title(f'X-Axis Vibration - Normal - {machine_id}')
        axes[0][1].plot(df_abnormal['x_axis'])
        axes[0][1].set_title(f'X-Axis Vibration - Abnormal - {machine_id}')
        
        axes[1][0].plot(df_normal['y_axis'])
        axes[1][0].set_title(f'Y-Axis Vibration - Normal - {machine_id}')
        axes[1][1].plot(df_abnormal['y_axis'])
        axes[1][1].set_title(f'Y-Axis Vibration - Abnormal - {machine_id}')
        
        axes[2][0].plot(df_normal['z_axis'])
        axes[2][0].set_title(f'Z-Axis Vibration - Normal - {machine_id}')
        axes[2][1].plot(df_abnormal['z_axis'])
        axes[2][1].set_title(f'Z-Axis Vibration - Abnormal - {machine_id}')
        
        plt.tight_layout()
        plt.show()


def read_hdf_files(normal_file_path, abnormal_file_path):
    normal_df = pd.read_hdf(normal_file_path)
    abnormal_df = pd.read_hdf(abnormal_file_path)
    return normal_df, abnormal_df

def read_hdf_files_random_test(normal_file_path, abnormal_file_path, n_normal=1, n_abnormal=1, verbose=False):
    # Get a list of all files in each directory
    normal_files = [f for f in os.listdir(normal_file_path) if f.endswith('.h5')]
    abnormal_files = [f for f in os.listdir(abnormal_file_path) if f.endswith('.h5')]
    
    # Randomly select 'n_normal' files from normal_files and 'n_abnormal' files from abnormal_files
    random_normal_files = random.sample(normal_files, n_normal)
    random_abnormal_files = random.sample(abnormal_files, n_abnormal)
    
    # Create DataFrames
    normal_dfs = {}
    abnormal_dfs = {}
    
    for file in random_normal_files:
        full_path = os.path.join(normal_file_path, file)
        normal_dfs[file] = pd.read_hdf(full_path)
        
    for file in random_abnormal_files:
        full_path = os.path.join(abnormal_file_path, file)
        abnormal_dfs[file] = pd.read_hdf(full_path)

    if (verbose):
        print(f'Randomized Normal files are {random_normal_files}')
        print(f'Randomized Abnormal files are {random_abnormal_files}')

    return normal_dfs, abnormal_dfs

def plot_vibration_data(df_normal, df_abnormal, savepath):
    axes = ['x_axis', 'y_axis', 'z_axis']
    time_normal = df_normal.index
    time_abnormal = df_abnormal.index

    fig, axs = plt.subplots(len(axes), 2, figsize=(14, 4*len(axes)))

    for idx, axis in enumerate(axes):
        # Plot vibration data for normal condition on the specific axis
        axs[idx, 0].plot(time_normal, df_normal[axis], color='b', linewidth=0.8)
        axs[idx, 0].set_title(f'Vibration Data Normal - {axis}', fontweight='bold')
        axs[idx, 0].set_xlabel('Time (s)')
        axs[idx, 0].set_ylabel('Amplitude')
        
        # Plot vibration data for abnormal condition on the specific axis
        axs[idx, 1].plot(time_abnormal, df_abnormal[axis], color='r', linewidth=0.8)
        axs[idx, 1].set_title(f'Vibration Data Abnormal - {axis}', fontweight='bold')
        axs[idx, 1].set_xlabel('Time (s)')
        axs[idx, 1].set_ylabel('Amplitude')

    plt.tight_layout(pad=2.0)
    plt.savefig(savepath + '/Vibration_Data_Plot.png', dpi=150)
    # plt.show()

def plot_combined_vibration_data(df_normal, df_abnormal, savepath):
    axes = ['x_axis', 'y_axis', 'z_axis']
    
    # Resetting the index for alignment
    time_normal = np.linspace(0, len(df_normal) * estimate_sampling_rate(df_normal), len(df_normal))
    time_abnormal = np.linspace(0, len(df_abnormal) * estimate_sampling_rate(df_abnormal), len(df_abnormal))

    alpha_value = 0.8
    fig, axs = plt.subplots(len(axes), 1, figsize=(10, 4*len(axes)))

    for idx, axis in enumerate(axes):
        # Plot vibration data for normal condition on the specific axis
        axs[idx].plot(time_normal, df_normal[axis], color='b', linewidth=0.8, label='Normal')
        
        # Plot vibration data for abnormal condition on the specific axis
        axs[idx].plot(time_abnormal, df_abnormal[axis], color='r', linewidth=0.8, label='Abnormal', alpha=alpha_value)
        
        axs[idx].set_title(f'Combined Vibration Data - {axis} axis', fontweight='bold')
        axs[idx].set_xlabel('Time (s)')
        axs[idx].set_ylabel('Amplitude')
        axs[idx].legend()

    plt.tight_layout(pad=2.0)
    plt.savefig(savepath + '/Combined_Vibration_Data_Plot.png', dpi=150)


def plot_psd(df_normal, df_abnormal, savepath):
    axes = ['x_axis', 'y_axis', 'z_axis']

    # Estimate sampling rates
    fs_normal = estimate_sampling_rate(df_normal)
    fs_abnormal = estimate_sampling_rate(df_abnormal)

    fig, axs = plt.subplots(len(axes), 2, figsize=(14, 4*len(axes)))

    for idx, axis in enumerate(axes):
        # Compute and plot PSD for normal condition on the specific axis
        freqs_normal, psd_normal = welch(df_normal[axis], fs=fs_normal)
        axs[idx, 0].semilogy(freqs_normal, psd_normal, color='b')
        axs[idx, 0].set_title(f'PSD Normal - {axis} axis')
        axs[idx, 0].set_xlabel('Frequency (Hz)')
        axs[idx, 0].set_ylabel('Power/Frequency (dB/Hz)')

        # Compute and plot PSD for abnormal condition on the specific axis
        freqs_abnormal, psd_abnormal = welch(df_abnormal[axis], fs=fs_abnormal)
        axs[idx, 1].semilogy(freqs_abnormal, psd_abnormal, color='r')
        axs[idx, 1].set_title(f'PSD Abnormal - {axis} axis')
        axs[idx, 1].set_xlabel('Frequency (Hz)')
        axs[idx, 1].set_ylabel('Power/Frequency (dB/Hz)')

    plt.tight_layout(pad=2.0)
    plt.savefig(savepath + '/PSD_Plot.png', dpi=150)

def plot_combined_psd(df_normal, df_abnormal, savepath):
    axes = ['x_axis', 'y_axis', 'z_axis']
   
    # Estimate sampling rates
    fs_normal = estimate_sampling_rate(df_normal)
    fs_abnormal = estimate_sampling_rate(df_abnormal)

    fig, axs = plt.subplots(len(axes), 1, figsize=(10, 4*len(axes)))

    for idx, axis in enumerate(axes):
        # Compute and plot PSD for normal condition on the specific axis
        freqs_normal, psd_normal = welch(df_normal[axis], fs=fs_normal)
        axs[idx].semilogy(freqs_normal, psd_normal, color='b', label='Normal')

        # Compute and plot PSD for abnormal condition on the specific axis
        freqs_abnormal, psd_abnormal = welch(df_abnormal[axis], fs=fs_abnormal)
        axs[idx].semilogy(freqs_abnormal, psd_abnormal, color='r', label='Abnormal')

        axs[idx].set_title(f'Combined PSD - {axis} axis')
        axs[idx].set_xlabel('Frequency (Hz)')
        axs[idx].set_ylabel('Power/Frequency (dB/Hz)')
        axs[idx].legend()

    plt.tight_layout(pad=2.0)
    plt.savefig(savepath + '/Combined_PSD_Plot.png', dpi=150)
    # plt.show()


def estimate_sampling_rate(df):
    n = len(df.index)
    if n < 2:
        raise ValueError("Dataframe needs at least two rows to estimate sampling rate.")
    
    # Calculate the average time difference over 100 points or less if the df has fewer points
    num_points = min(n, 100) # the 'datetime' column is set as the df's index, difference is milliseconds.
    time_diff_in_seconds = (df.index[-1] - df.index[0]).total_seconds()  # Convert time difference to seconds.
    average_time_diff_in_seconds = time_diff_in_seconds / (num_points - 1)
    
    # Estimate the sample rate
    sample_rate = 1 / average_time_diff_in_seconds
    return sample_rate

def plot_fft(df1, df2, savepath):
    # Compute FFT for each axis
    fft_axes = ['x_axis', 'y_axis', 'z_axis']
    colors = {'x_axis': '#FF1F5B', 'y_axis': '#009ADE', 'z_axis': '#FFC61E'} # ["#FF1F5B", "#009ADE", "#FFC61E"]
    fft_normal = {axis: compute_fft(df1[axis]) for axis in fft_axes}
    fft_abnormal = {axis: compute_fft(df2[axis]) for axis in fft_axes}

    # Initialize the figure
    plt.figure(figsize=(18, 12))

    # Create subplots for each axis
    for i, axis in enumerate(fft_axes, 1):
        plt.subplot(3, 2, 2*i - 1)
        plt.title(f'FFT Normal {axis}', fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.plot(np.abs(fft_normal[axis])[:len(fft_normal[axis])//2], label=axis, color=colors[axis])  # Plot only the positive freqs up to Nyquist freq   
        plt.legend(loc='upper right')

        plt.subplot(3, 2, 2*i)
        plt.title(f'FFT Abnormal {axis}', fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.plot(np.abs(fft_abnormal[axis])[:len(fft_abnormal[axis])//2], label=axis, color=colors[axis])  # Plot only the positive freqs up to Nyquist freq  
        plt.legend(loc='upper right')

    plt.subplots_adjust(hspace=0.5) # Adjust vertical spacing    
    plt.savefig(savepath + '/FFT_Plot.png') # Save the plot
    # plt.draw()

def custom_colormap():
    colors = [(1, 1, 0), (1, 0, 0)]  # Yellow to Red
    return LinearSegmentedColormap.from_list("yellowRed", colors)

def plot_spectrogram(df1, df2, savepath, nperseg_val=256):
    plt.figure(figsize=(14, 8))
    
    # Estimate sampling rates
    sampling_rate1 = estimate_sampling_rate(df1)
    sampling_rate2 = estimate_sampling_rate(df2)
    
    # Titles
    title1 = 'Spectrogram - Normal'
    title2 = 'Spectrogram - Abnormal'
    
    axes = ['x_axis', 'y_axis', 'z_axis']
    high_contrast_cmap = 'jet'
    my_cmap = custom_colormap()

    # This is 
    for i, axis in enumerate(axes, 1):
        plt.subplot(3, 2, 2*i-1)
        plt.title(f'{title1} {axis}', fontweight='bold')
        f1, t1, Sxx1 = spectrogram(df1[axis], sampling_rate1, nperseg=nperseg_val)
        im1 = plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud', cmap=high_contrast_cmap)
        plt.yscale('log')
        plt.ylim(f1[0], f1[-1])
        plt.ylabel('Frequency [Hz] - logarithmic')
        plt.xlabel('Time [sec]')
        clb1 = plt.colorbar(im1)
        clb1.set_label('Magnitude [dB]', rotation=90, labelpad=10)
        
        plt.subplot(3, 2, 2*i)
        plt.title(f'{title2} {axis}', fontweight='bold')
        f2, t2, Sxx2 = spectrogram(df2[axis], sampling_rate2, nperseg=nperseg_val)
        im2 = plt.pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud', cmap=high_contrast_cmap)
        plt.yscale('log')
        plt.ylim(f2[0], f2[-1])
        plt.ylabel('Frequency [Hz] - logarithmic')
        plt.xlabel('Time [sec]')
        clb2 = plt.colorbar(im2)
        clb2.set_label('Magnitude [dB]', rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(savepath + '/Spectrogram_Plot.png', dpi=150)
    # plt.show()

def plot_stft(df_normal, df_abnormal, savepath):
    axes = ['x_axis', 'y_axis', 'z_axis']
    my_cmap = 'virdis'
    
    # Adjusting the figure size for better visualization
    fig, axs = plt.subplots(len(axes), 2, figsize=(14, 4*len(axes)))

    for idx, axis in enumerate(axes):
        # Compute STFT for the specific axis
        frequencies_normal, times_normal, Zxx_normal = compute_stft(df_normal[axis])
        frequencies_abnormal, times_abnormal, Zxx_abnormal = compute_stft(df_abnormal[axis])

        # Plot STFT for normal data on the specific axis
        im1 = axs[idx, 0].imshow(np.abs(Zxx_normal), aspect='auto', cmap='viridis', 
                                 extent=[times_normal.min(), times_normal.max(), frequencies_normal.min(), 
                                         frequencies_normal.max()])
        axs[idx, 0].set_title(f'STFT Normal - {axis}', fontweight='bold')
        axs[idx, 0].set_xlabel('Time (s)')
        axs[idx, 0].set_ylabel('Frequency (Hz) - logarithmic')
        axs[idx, 0].set_yscale('log')
        axs[idx, 0].set_ylim(frequencies_normal.min(), frequencies_normal.max()) # Set y-axis limits
        clb1 = fig.colorbar(im1, ax=axs[idx, 0], format='%+2.0f dB') # Add colorbar
        clb1.set_label('Magnitude [dB]', rotation=90, labelpad=10)

        # Plot STFT for abnormal data on the specific axis
        im2 = axs[idx, 1].imshow(np.abs(Zxx_abnormal), aspect='auto', cmap='viridis', 
                                 extent=[times_abnormal.min(), times_abnormal.max(), frequencies_abnormal.min(), 
                                         frequencies_abnormal.max()])
        axs[idx, 1].set_title(f'STFT Abnormal - {axis}', fontweight='bold')
        axs[idx, 1].set_xlabel('Time (s)')
        axs[idx, 1].set_ylabel('Frequency (Hz) - logarithmic')
        axs[idx, 1].set_yscale('log')
        axs[idx, 1].set_ylim(frequencies_abnormal.min(), frequencies_abnormal.max()) # Set y-axis limits
        clb2 = fig.colorbar(im2, ax=axs[idx, 1], format='%+2.0f dB') # Add colorbar
        clb2.set_label('Magnitude [dB]', rotation=90, labelpad=10)

    plt.tight_layout(pad=2.0) # Added padding to further reduce whitespace
    plt.savefig(savepath + '/STFT_Plot.png', dpi=150)
    # plt.show()

def compute_fft(data):
    return fft(data.values.flatten())

def compute_stft(data):
    return stft(data.values.flatten())

def plot_vibration_fft_and_stft(normal_df, abnormal_df, vibration_plot=True, fft_plot=False, spectrogram_plot=False, stft_plot=False, psd_plot=False):
    
    save_path='plots'
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Raw Vibration Plot - normal and abnormal data
    if (vibration_plot):
        # plot_vibration_data(normal_df, abnormal_df, save_path) # separate plots
        plot_combined_vibration_data(normal_df, abnormal_df, save_path) # normal and abnormal overlay

    # FFT Plot - normal and abnormal data
    if (fft_plot):
        plot_fft(normal_df, abnormal_df, save_path)

    # Spectrogram Plot - normal and abnormal data
    if (spectrogram_plot):
        plot_spectrogram(normal_df, abnormal_df, save_path)

    # STFT Plot - normal and abnormal data
    if (stft_plot):
        plot_stft(normal_df, abnormal_df, save_path)

    # Power Spectral Density Plot - normal and abnormal data
    if (psd_plot):
        # plot_psd(normal_df, abnormal_df, save_path)
        plot_combined_psd(normal_df, abnormal_df, save_path)


def plot_side_by_side(normal_df, abnormal_df):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Normal')
    plt.plot(normal_df)
    plt.subplot(1, 2, 2)
    plt.title('Abnormal')
    plt.plot(abnormal_df)
    plt.legend()
    plt.show()

def plot_side_by_side_plotly(normal_df, abnormal_df):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Normal", "Abnormal"))

    # Plotting data for Normal
    fig.add_trace(go.Scatter(x=normal_df.index, y=normal_df['x_axis'], mode='lines', name='X', showlegend=True, opacity=0.6, line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=normal_df.index, y=normal_df['y_axis'], mode='lines', name='Y', opacity=0.6, showlegend=True, line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=normal_df.index, y=normal_df['z_axis'], mode='lines', name='Z', opacity=0.6, showlegend=True, line=dict(color='blue')), row=1, col=1)
    
    # Plotting data for Abnormal
    fig.add_trace(go.Scatter(x=abnormal_df.index, y=abnormal_df['x_axis'], mode='lines', name='X', showlegend=True, line=dict(color='red'), opacity=0.6), row=1, col=2)
    fig.add_trace(go.Scatter(x=abnormal_df.index, y=abnormal_df['y_axis'], mode='lines', name='Y', showlegend=True, line=dict(color='green'), opacity=0.6), row=1, col=2)
    fig.add_trace(go.Scatter(x=abnormal_df.index, y=abnormal_df['z_axis'], mode='lines', name='Z', showlegend=True, line=dict(color='blue'), opacity=0.6), row=1, col=2)

    # Update y-axis range and legends
    fig.update_yaxes(range=[-4000, 4000], row=1, col=1)
    fig.update_yaxes(range=[-4000, 4000], row=1, col=2)

    # Update layout
    fig.update_layout(
        title='Comparison of Normal and Abnormal Data',
        xaxis_title='Datetime', 
        title_x=0.5,
        yaxis_title='Vibration Amplitude'
    )
    
    fig.show()

def create_save_pairplot(df, title, filename):
    """
    This function creates and saves a pairplot.
    
    Parameters:
    - df: DataFrame containing the data to plot.
    - title: Title for the plot.
    - filename: Name of the file to save the plot.
    """
    pairplot = sns.pairplot(df)
    plt.suptitle(title)
    pairplot.savefig(f'plots/{filename}.png')
    plt.close()

def create_save_combined_pairplot(normal_df, abnormal_df, filename):
    """
    This function combines two datasets, creates a binary 'Status' label, 
    creates a pairplot, and saves it.
    
    Parameters:
    - normal_df: DataFrame containing the normal data.
    - abnormal_df: DataFrame containing the abnormal data.
    - filename: Name of the file to save the plot.
    """
    # Combine the datasets with a binary 'Status' column
    normal_df['Status'] = 0  # Binary label for normal
    abnormal_df['Status'] = 1  # Binary label for abnormal
    combined_df = pd.concat([normal_df, abnormal_df])
    
    # Generate the pairplot
    pairplot = sns.pairplot(combined_df, 
                            hue='anomaly',
                            diag_kind='kde', # kernel density estimation
                            diag_kws={'color':'red'},
                            kind = "reg"
    ) #palette={0: "blue", 1: "red"})
                            # kind="reg", # regression
                            # palette={0: "blue", 1: "red"})

    plt.suptitle('Combined Pairplot of Vibration Data')
    pairplot.savefig(f'plots/{filename}_combined.png')
    plt.close()

def create_save_combined_pairplot_test(normal_dfs, abnormal_dfs, filename, verbose=False):
    """
    This function combines multiple datasets from dictionaries into one DataFrame,
    creates a pairplot, and saves it.
    
    Parameters:
    - normal_dfs: Dictionary of DataFrames containing the normal data.
    - abnormal_dfs: Dictionary of DataFrames containing the abnormal data.
    - filename: Name of the file to save the plot.
    """
    # Initialize lists to hold the concatenated DataFrames
    normal_data = []
    abnormal_data = []
    
    # Iterate over the normal datasets and concatenate them
    for key, df in normal_dfs.items():
        df['anomaly'] = 0  # Binary label for normal
        normal_data.append(df)

    # Iterate over the abnormal datasets and concatenate them
    for key, df in abnormal_dfs.items():
        df['anomaly'] = 1  # Binary label for abnormal
        abnormal_data.append(df)
    
    if (verbose): print('df tagging completed')
    
    # Combine all normal and abnormal DataFrames
    combined_normal_df = pd.concat(normal_data)
    combined_abnormal_df = pd.concat(abnormal_data)

    # Combine the two sets into one DataFrame
    combined_df = pd.concat([combined_normal_df, combined_abnormal_df])
    
    if (verbose): print('Starting pairplot')

    # Generate the pairplot
    sns.pairplot(combined_df, 
                            hue='anomaly',
                            markers=["o", "^"], 
                            palette= "pastel", #{0: "#a6cee3", 1: "#ff7f00"},
                            diag_kind='kde', # kernel density estimation
                            kind="reg", # regression lines
                            plot_kws={"line_kws":{"color":"red"}}
    )
    
    plt.suptitle('Combined Pairplot of Machine Data', y=1.02)
    plt.savefig(f'plots/{filename}_combined.png')
    plt.close()

def plot_evaluation_curves(y_test, pred_proba):
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, pred_proba)
    pr_auc = auc(recall, precision)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.getcwd() + '\\plots\AURROC_AUPRC.png', dpi=150)
    # plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.getcwd() + '\\plots\\confusion_matrix.png', dpi=150)
    # plt.show()

def anomaly_detection(X_train, y_train, X_test, y_test, algorithm='All', en_feature_eng=False, en_hyperparam_tune=False):
    result = {} # store results for each algos (if applicable)
    pred_proba_dict = {} # store prod_proba values for each algo.
    m_cpus = os.cpu_count()

    if not en_feature_eng: # and len(X_train.shape) > 2:
        X_train = X_train.reshape(-1, X_train.shape[-1])
        X_test = X_test.reshape(-1, X_test.shape[-1])
        print("Reshaped X_train:", X_train.shape)
        print("Reshaped X_test:", X_test.shape)
    
    print("Feature Engineering enabled:", en_feature_eng)

    def evaluate_model(y_test, pred, pred_proba=None):
        '''
        NOTE: 
            AUPRC: Focused view on the minority class peformance
            AUROC: Holistic view on the model's capabilities to distinguish between positive and negative classes.
        '''
        return {
            'f1_score': f1_score(y_test, pred),
            'precision': precision_score(y_test, pred),
            'recall': recall_score(y_test, pred),
            'auprc': average_precision_score(y_test, pred_proba), # AUPRC
            'auroc': roc_auc_score(y_test, pred_proba)
        }
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_samples = trial.suggest_float('max_samples', 0.1, 1)
        max_features = trial.suggest_float('max_features', 0.1, 1)
        contamination = trial.suggest_float('contamination', 0, 0.5)

        print(f'The n_estimator value is: {n_estimators}')
        print(f'The max_samples value is: {max_samples}')
        print(f'The max_features value is: {max_features}')
        print(f'The contamination value is: {contamination}')
        
        model = IsolationForest(n_estimators=n_estimators,
                                max_samples=max_samples,
                                max_features=max_features,
                                contamination=contamination,
                                n_jobs=int(m_cpus-4))
        print('IF Model fitting')
        model.fit(X_train)
        print('IF Model Predicting')
        pred = model.predict(X_test)
        pred[pred == 1] = 0
        pred[pred == -1] = 1

        # Predict probabilities and assign an anomaly score for each sample
        pred_proba = model.decision_function(X_test)
        
        # precision, recall, _ = precision_recall_curve(y_test, pred_proba)
        # auprc = auc(recall, precision)

        f1 = f1_score(y_test, pred, pos_label=1)
        recall = recall_score(y_test, pred, pos_label=1)
        precision = precision_score(y_test, pred, pos_label=1)
        auprc = average_precision_score(y_test, pred)
        
        return f1, recall, precision, auprc
    
    # Custom sorting function to sort first by F1 score (index 0) and then by precision (index 1)
    def custom_sort(trial):
        return (trial.values[0], trial.values[1])

    if algorithm == 'IsolationForest':
        print("\n--- Running Isolation Forest ---")

        if (en_hyperparam_tune):
            print("Running Hyper-parameter search first")
            study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize']) 
            # study = optuna.create_study(directions=['maximize']) # single objective
            study.optimize(objective, n_trials=10)
            
            # best_params = study.best_params # This is only applicable for single objective optimization only.

            # Multiple objectives - Sort trials
            # -------------
            sorted_best_trials = sorted(study.best_trials, key=custom_sort, reverse=True)  # Sorting by multiple objectives

            # Choose the trial that gives the maximum F1 score and then maximum precision in case of ties
            best_trial = sorted_best_trials[0]
            
            best_params = best_trial.params
            # Print the best parameters only at the end
            print(f"Best parameters are: {best_params}")
            # -------------

            model = IsolationForest( n_jobs=int(m_cpus-4), **best_params) # Only for hyper-parameter fine-tuning.
        else:
            # --- Config 2
            model = IsolationForest(n_estimators=156, max_samples=0.54545,            
                                    max_features=0.88992, contamination=0.23176, 
                                    n_jobs=int(m_cpus-4)) # Use this after hyperparameter has been established.
            # model = IsolationForest(n_estimators=2, n_jobs=int(m_cpus-4)) # Baseline configuration
            # --- Config 1
            # model = IsolationForest(n_estimators=138, max_samples=0.8554,            
            #                         max_features=0.5411, contamination=0.4441, 
            #                         n_jobs=int(m_cpus-4)) # Use this after hyperparameter has been established.
            
        print('IF Model fitting')
        model.fit(X_train)
        print('IF Model Predicting')
        pred = model.predict(X_test)
        pred[pred == 1] = 0
        pred[pred == -1] = 1
        pred_proba = model.decision_function(X_test)

        # Plotting the confusion matrix
        classes = [0, 1]  # or ['normal', 'anomaly']
        plot_confusion_matrix(y_test, pred, classes)
    
    elif algorithm == 'LSTM-Autoencoder':
        print("\n--- Running LSTM Autoencoder ---")
        # Assuming your data is reshaped appropriately for LSTM input
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            LSTM(32, activation='relu', return_sequences=False),
            RepeatVector(X_train.shape[1]),
            LSTM(32, activation='relu', return_sequences=True),
            LSTM(64, activation='relu', return_sequences=True),
            TimeDistributed(Dense(X_train.shape[2]))
        ])

        # Initialize callbacks
        dynamic_threshold_callback = DynamicThresholdCallback()
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
        tensorboard = TensorBoard(log_dir='./logs')

        model.compile(optimizer='adam', loss='mse')        
        model.fit(X_train, X_train, epochs=50, batch_size=128, validation_split=0.1, callbacks=[dynamic_threshold_callback, model_checkpoint, tensorboard])
        
        # Calculate dynamic threshold
        dynamic_threshold = dynamic_threshold_callback.get_dynamic_threshold()
        
        pred_reconstructions = model.predict(X_test)
        # Transform pred to match y_test, e.g., using a threshold on the reconstruction error
        mse = np.mean(np.square(pred_reconstructions - X_test), axis=1)

        en_dynamic_thresholding = True
        if (en_dynamic_thresholding):
            pred = (mse > dynamic_threshold).astype(int)    
        else:
            threshold = 0.2 # Fixed to determien suitable threshold
            pred = (mse > threshold).astype(int)

        pred_proba = mse # Use the reconstruction error as pred_proba

    print('Test set Classification Report is: ')
    print("Shape of y_test:", y_test.shape)
    print("Shape of pred:", pred.shape)
    print(classification_report(y_test, pred))
    pred_proba_dict[algorithm] = pred_proba
    print('Evaluating model under test')
    result[algorithm] = evaluate_model(y_test, pred, pred_proba=pred_proba)
    
    # Visualize the ROC and Precision-Recall curves
    plot_evaluation_curves(y_test, pred_proba)

    return result


def plot_sequence_comparison(normal_sequences, abnormal_sequences, X_train, y_train):
    """
    Plot comparisons between original sequences and the training set.
    
    Parameters:
    - normal_sequences: list of np.arrays, preprocessed normal sequences
    - abnormal_sequences: list of np.arrays, preprocessed abnormal sequences
    - X_train: np.array, the training data
    - y_train: np.array, the training labels
    """

    # Ensure the normal and abnormal sequences are numpy arrays
    normal_sequences = np.array(normal_sequences)
    abnormal_sequences = np.array(abnormal_sequences)
    
    # Extract the sequence length and number of features from the normal sequences
    try:
        seq_length, features = normal_sequences.shape[1], normal_sequences.shape[2]
    except IndexError:
        raise ValueError("Normal sequences do not have the required dimensions (samples, time_steps, features).")
    
    print(f"Shapes - Normal: {normal_sequences.shape}, Abnormal: {abnormal_sequences.shape}, X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Initial X_train.ndim: {X_train.ndim}")
    
    # Reshape X_train if it is flattened
    if X_train.ndim == 1:
        print("Attempting to reshape X_train.")
        # Calculate the total number of expected features (time_steps * features per time step)
        total_features = seq_length * features
        
        # Check if the total size of X_train is divisible by the total number of expected features
        if X_train.size % total_features == 0:
            num_samples = X_train.size // total_features
            # Reshape X_train to be three-dimensional: (samples, time_steps, features)
            X_train = X_train.reshape((num_samples, seq_length, features))
        else:
            # If the size is not divisible, raise an error to indicate that the reshape cannot be performed
            raise ValueError(f"Cannot reshape X_train to be ({num_samples}, {seq_length}, {features}). The total size of X_train ({X_train.size}) is not divisible by the total number of features ({total_features}).")
        
        print(f"Reshaped X_train.shape: {X_train.shape}")

    # Define the number of samples to plot
    num_samples_to_plot = min(3, normal_sequences.shape[0], abnormal_sequences.shape[0], X_train.shape[0])

    fig, axs = plt.subplots(num_samples_to_plot, 3, figsize=(15, num_samples_to_plot * 5))

    for i in range(num_samples_to_plot):
        axs[i, 0].plot(normal_sequences[i][:, 0], label='Normal')
        axs[i, 1].plot(abnormal_sequences[i][:, 0], label='Abnormal')
        
        # Check if X_train has been reshaped correctly
        if X_train.ndim == 3:
            axs[i, 2].plot(X_train[i][:, 0], label=f'Trained {"Normal" if y_train[i] == 0 else "Abnormal"}')
        else:
            print(f"Cannot plot for sample {i} as X_train is not 3-dimensional.")

        for j in range(3):
            axs[i, j].legend()

    plt.tight_layout()
    plt.show()
