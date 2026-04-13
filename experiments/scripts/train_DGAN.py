import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import argparse

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import normalize

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, Normalization

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft
from scipy.stats import ks_2samp

import quality_metrics as data_qm
import torch

from torch.utils.tensorboard import SummaryWriter


#===================================================================
#=============== 00. List of Globals  ==============================
#===================================================================
failure_type = 'Ball_Screw'
df = pd.DataFrame()

parser = argparse.ArgumentParser(description='Train DoppelGANger model on RBCC vibration dataset')

#List of Flags
parser.add_argument(                               '--gpu_id', type=int,   default=1,  help='Select which GPU to run the DoppelGANger training on')
parser.add_argument(              '--clean_data_from_scratch', type=int,   default=0,  help='Flag for concatenating all h5 files of RBCC abnormal data + Standardizing data')
parser.add_argument(                      '--data_imputation', type=int,   default=0,  help='Flag for imputing data to ensure each example_id has same sequence length')
parser.add_argument(                                '--train', type=int,   default=1,  help='Flag for training DGAN model')
parser.add_argument(                   '--generate_synthetic', type=int,   default=1,  help='Flag for generating synthetic data based on a trained model')
parser.add_argument(                             '--evaluate', type=int,   default=1,  help='Flag for performing evaluation')

#Section 01 Globals
df = pd.DataFrame()
max_sequence_len = 0
cols_to_drop = ['date', 'time']
abnormal_path_dict = {'Ball_Screw'     : '../data/AHC-A3/abnormal/Ball_screw/Ball_screw_220916',
                      'BartoBarFailure': '../data/AHC-A3/abnormal/BartoBarFailure/BartoBarFailure_20220328',
                      'Belt_wheel'     : '../data/AHC-A3/abnormal/Belt_wheel/Belt_wheel_220228/',
                     }
threshold_seq_len    = 150000

#Section 02 Globals
feature_cols = ['x_axis', 'y_axis', 'z_axis']
convert_cols = ['Sts'] + feature_cols
example_id_column = 'example_id'

#Section 03 Globals
num_imputation_repeat = 1000

#Section 04 Globals
parser.add_argument(                               '--exp_id', type=str,   default='Exp8',help='Divisor of max_sequence_len. It is advised to be between 10 and 20. Must be a factor of 118530')
parser.add_argument(                           '--sample_len', type=int,   default=2634,  help='Divisor of max_sequence_len. It is advised to be between 10 and 20. Must be a factor of 118530')
parser.add_argument(                           '--batch_size', type=int,   default=  18,  help='Recommended value in gretel-synthetics package is 1000, but will need to adjust based on computing power')
parser.add_argument(                               '--epochs', type=int,   default=1500,  help='For large dataset, it is appropriate to set epochs between 100 to 1000')
parser.add_argument(                 '--attribute_num_layers', type=int,   default=  16,  help='Number of Discriminator network\'s hidden layers') 
parser.add_argument(                  '--attribute_num_units', type=int,   default= 512,  help='Number of Discriminator network\'s hidden nodes') 
parser.add_argument(                   '--feature_num_layers', type=int,   default=  16,  help='Number of Generator     network\'s hidden layers')
parser.add_argument(                    '--feature_num_units', type=int,   default= 512,  help='Number of Generator     network\'s hidden layers')
parser.add_argument(              '--generator_learning_rate', type=float, default=1e-4,  help='Adam learning rate for Generator     network')
parser.add_argument(          '--discriminator_learning_rate', type=float, default=1e-4,  help='Adam learning rate for Discriminator network')
parser.add_argument('--attribute_discriminator_learning_rate', type=float, default=1e-4,  help='Adam learning rate for Auxiliary Discriminator network')
parser.add_argument(                '--apply_feature_scaling', type=bool,  default=False, help='Scale continuous variables inside the model. Can only set to False if inputs are already scaled to [0, 1] or [-1, 1] ')
parser.add_argument(                 '--discriminator_rounds', type=int,   default=   3,  help='Number of training steps per batch for Discriminator')
parser.add_argument(                     '--generator_rounds', type=int,   default=   1,  help='Number of training steps per batch for Generator')
parser.add_argument(                       '--epoch_patience', type=int,   default=   1,  help='Tolerance of maximum consecutive epoch with absolute delta loss values less than delta_loss_patience')
parser.add_argument(                  '--delta_loss_patience', type=float, default=1e-3,  help='Minimum threshold for delta loss value that Generator or Discriminator networks must have within epoch window equals to epoch_patience')
parser.add_argument(             '--num_epoch_per_checkpoint', type=int,   default=   2,  help='Saving checkpoint states for every 2 epochs in training by default. Any abort event will avoid from training from epoch 0 all over again.')
parser.add_argument(                        '--oscillate_DLR', type=bool,  default= True, help='Implement varying learning rate on Discriminator network')
parser.add_argument(                        '--oscillate_GLR', type=bool,  default=False, help='Implement varying learning rate on Generator     network')
parser.add_argument(               '--KLdiv_loss_replacement', type=bool,  default=False, help='Replace loss_generated and loss_real with KL divergence loss in Discriminator\'s loss function within _train() function in gretel-synthetics\' dgan.py file')
parser.add_argument(                  '--KLdiv_loss_addition', type=bool,  default= True, help='Add KL divergence loss in time domain as the new components in in Discriminator\'s loss function within _train() function in gretel-synthetics\' dgan.py file')
parser.add_argument(              '--KLdiv_fft_loss_addition', type=bool,  default=False, help='Add KL divergence loss in freq domain as the new components in in Discriminator\'s loss function within _train() function in gretel-synthetics\' dgan.py file')
parser.add_argument(               '--spectral_loss_addition', type=bool,  default=False, help='Add weighted MSE FFT loss across the frequency spectrum')

args = parser.parse_args()
print(f'\n\nPrinting all script arguments:\n')
for arg_val in vars(args):
    print(f'{arg_val}: {getattr(args, arg_val)}')
print('\n\n')

SLname   = '_SL'   + str(args.sample_len)
BSname   = '_BS'   + str(args.batch_size)
Ename    = '_E'    + str(args.epochs)
ANLname  = '_ANL'  + str(args.attribute_num_layers)
ANUname  = '_ANU'  + str(args.attribute_num_units)
FNLname  = '_FNL'  + str(args.feature_num_layers)
FNUname  = '_FNU'  + str(args.feature_num_units)
GLRname  = '_GLR'  + str('{:.1e}'.format(args.generator_learning_rate))
DLRname  = '_DLR'  + str('{:.1e}'.format(args.discriminator_learning_rate))
ADLRname = '_ADLR' + str('{:.1e}'.format(args.attribute_discriminator_learning_rate))

#List of filenames
cwd_path = '/home/mluser_intern/Desktop/rbcc_git/dgan/'  # All filepaths in this script is specified wrt this cwd_path
target_minmax_offset_filename = 'df_minmax.csv' #This minmax_offset file is used to stretch back synthetic_df to get a signal the more resemble raw data
target_h5_real_filename       = 'clean_data.h5'
target_h5_impt_filename       = 'clean_imputed_data.h5'
target_model                  = SLname + BSname + Ename + ANLname + ANUname + FNLname + FNUname + GLRname + DLRname + ADLRname
target_model_filename         = target_model + '.model'
target_h5_synt_filename       = target_model + '_synt_data.h5'
target_metric_filename        = target_model + '_metrics.log'

target_minmax_offset_filepath = cwd_path + target_minmax_offset_filename
target_h5_real_filepath       = cwd_path + target_h5_real_filename               #If you don't have this file yet, please change variable clean_data_from_scratch to 1 and rerun this script again.
target_h5_impt_filepath       = cwd_path + target_h5_impt_filename               #If you don't have this file yet, please change variable clean_data_from_scratch to 1 and rerun this script again.
target_model_filepath         = cwd_path + args.exp_id + '_' + target_model_filename 
target_h5_synt_filepath       = cwd_path + args.exp_id + '_' + target_h5_synt_filename               
target_metric_filepath        = cwd_path + args.exp_id + '_' + target_metric_filename
target_tensorboard_filepath   = cwd_path + 'tensorboard/' + args.exp_id + '_' + target_model

print(f'\n\ntarget_model_filepath:\n{target_model_filepath}\n\n')

print(f'\n\nTensorBoard CMD: tensorboard --logdir={target_tensorboard_filepath}\n\n')


#===================================================================
#=============== 00. User-defined Functions ========================
#===================================================================
def set_gpu_device(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"]       = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]    = str(gpu_id) 
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=10"


def get_date_only(x):
    return str(x).split(' ')[0]


def populate_each_sampleID(df, num_repeat, groupby_colname):
    df_list = []

    for i in tqdm(range(len(df[groupby_colname].unique().tolist())), desc="Populating_each_sample_ID Part 1...", ascii=False, ncols=75):
        df_list += [df[df[groupby_colname]==i].reset_index(drop=True)]

    max_sequence = 0
    for i in tqdm(range(len(df_list)), desc="Populating_each_sample_ID Part 2...", ascii=False, ncols=75):
        max_sequence = max(max_sequence, df_list[i].shape[0])


    for i in tqdm(range(len(df_list)), desc="Populating_each_sample_ID Part 3...", ascii=False, ncols=75):
        n = df_list[i].shape[0]
        quotient   = (max_sequence-n) // num_repeat
        remainder  = (max_sequence-n) %  num_repeat
        df_list[i] = pd.concat([df_list[i]] + [df_list[i][-num_repeat:]]*quotient + [df_list[i][-num_repeat:-num_repeat+remainder]], axis=0, ignore_index=True)
    
    temp = pd.DataFrame()
    for i in tqdm(range(len(df_list)), desc="Populating_each_sample_ID Part 4...", ascii=False, ncols=75):
        temp = pd.concat([temp, df_list[i]], ignore_index=True)

    del df_list
    return max_sequence, temp


def plot_histogram(real_data, synthetic_data, axis_name, compare_by_sts=False): #Unified histogram plotting function with statistical metrics
    def add_stats_to_plot(real_data, synthetic_data, axis_name, ax):
        #Calculate statistics
        mean_real, std_real = real_data.mean(), real_data.std()
        mean_synthetic, std_synthetic = synthetic_data.mean(), synthetic_data.std()
        ks_stat, ks_p = ks_2samp(real_data, synthetic_data)

        #Display statistics on plot
        ax.text(0.02, 0.95, f'Real : mean={mean_real:.2f}, std={std_real:.2f}', transform=ax.transAxes)
        ax.text(0.02, 0.90, f'Synth: mean={mean_synthetic:.2f}, std={std_synthetic:.2f}', transform=ax.transAxes)
        ax.text(0.02, 0.85, f'K-S Test: stat={ks_stat:.2f}, p={ks_p:.2f}', transform=ax.transAxes)

    if compare_by_sts:
        sts_values = real_data['Sts'].unique()

        for sts in sts_values:
            fig, ax = plt.subplots(figsize=(10, 4))

            #Filter data by 'Sts' value
            filtered_real_data = real_data[real_data['Sts']==sts][axis_name]
            filtered_synthetic_data = synthetic_data[synthetic_data['Sts']==sts][axis_name]

            ax.hist(filtered_real_data, bins=30, alpha=0.5, label=f'Real Data (Sts={sts})')
            ax.hist(filtered_synthetic_data, bins=30, alpha=0.5, label=f'Synthetic Data (Sts={sts})')

            add_stats_to_plot(filtered_real_data, filtered_synthetic_data, axis_name, ax)

            ax.set_xlabel(f'{axis_name} Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Comparison on {axis_name} Axis (Sts={sts})')
            ax.legend()
            #plt.show()
    else:
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.hist(real_data[axis_name], bins=30, alpha=0.5, label='Real Data')
        ax.hist(synthetic_data[axis_name], bins=30, alpha=0.5, label='Synthetic Data')

        add_stats_to_plot(real_data[axis_name], synthetic_data[axis_name], axis_name, ax)

        ax.set_xlabel(f'{axis_name} Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Comparison on {axis_name} Axis')
        ax.legend()
        #plt.show()
    plt.savefig(cwd_path + args.exp_id + '_' + target_model + '_histo.JPG')


def add_gaussian_noise(df, std_dev, columns): #Function to add Gaussian noise to a DataFrame
    print("\nEntering add_gaussian_noise() function...\n")

    #Copy the DataFrame to avoid modifying the original 
    df_noisy = df.copy()

    #Add Gaussian noise to each specified column 
    for col in tqdm(columns, desc="add_gaussian_noise...", ascii=False, ncols=75):
        noise = np.random.normal(0, std_dev, df_noisy[col].shape)
        df_noisy[col] += noise

    print("\nExiting  add_gaussian_noise() function...\n")
    return df_noisy


def extended_plot_distributions(real_df, synthetic_df, gauss_df, title):
    print("\nEntering extended_lot_distributions() function...\n")
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12,9))

    for i, axis in tqdm(enumerate(feature_cols), desc="extended_plot_distributions...", ascii=False, ncols=75):
        #Real vs Synthetic
        sns.kdeplot(real_df[axis], ax=axes[i, 0], color='blue', fill=True, label='Real Data')
        sns.kdeplot(synthetic_df[axis], ax=axes[i, 0], color='green', fill=True, label='Synthetic Data')
        axes[i, 0].set_title(f'Distribution for {axis} (Real vs Synthetic)')
        axes[i, 0].legend()

        #Real vs Gaussian
        sns.kdeplot(real_df[axis], ax=axes[i, 1], color='blue', fill=True, label='Real Data')
        sns.kdeplot(gauss_df[axis], ax=axes[i, 1], color='red', fill=True, label='Gaussian Data')
        axes[i, 1].set_title(f' Distribution for {axis} (Real vs Gaussian)')
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.savefig(cwd_path + args.exp_id + '_' + target_model + '_' + title + '.JPG')
    print("\nExiting  extended_plot_distributions() function...\n")
    #plt.show()


def side_by_side_comparison(real_df, synthetic_df, gauss_df, comparison_function, title, percentage=0.5): #, n_samples=500):
    print("\nEntering side_by_side_comparison() function...\n")
    n_samples = int(percentage*len(df)) #Use 40% of the datasets for analysis.
    real_df_sample = real_df[feature_cols].head(n_samples)
    synthetic_df_sample = synthetic_df[feature_cols].head(n_samples)
    gauss_df_sample = gauss_df[feature_cols].head(n_samples)

    num_axis = len(feature_cols)
    fig, axes = plt.subplots(num_axis, 3, figsize=(15, 5*num_axis))


    for i, axis in tqdm(enumerate(feature_cols), desc="side_by_side_comparison...", ascii=False, ncols=75):
        #Real Data
        ax = plt.subplot(num_axis, 3, i*3 + 1)
        if comparison_function==plot_pairwise_distances:
            comparison_function(real_df_sample[[axis]])
        else:
            comparison_function(real_df_sample[[axis]], 'blue', 'Real Data')
        plt.title(f'Real Data ({axis}) - {title}')

        #Synthetic Data
        ax = plt.subplot(num_axis, 3, i*3 + 2)
        if comparison_function==plot_pairwise_distances:
            comparison_function(synthetic_df_sample[[axis]])
        else:
            comparison_function(synthetic_df_sample[[axis]], 'green', 'Synthetic Data')
        plt.title(f'Synthetic Data ({axis}) - {title}')

        #Gaussian Data
        ax = plt.subplot(num_axis, 3, i*3 + 3)
        if comparison_function==plot_pairwise_distances:
            comparison_function(gauss_df_sample[[axis]])
        else:
            comparison_function(gauss_df_sample[[axis]], 'red', 'Gaussian Data')
        plt.title(f'Gaussian Data ({axis}) - {title}')

    plt.tight_layout()
    plt.savefig(cwd_path + args.exp_id + '_' + target_model + '_' + title + '.JPG')
    print("\nExiting  side_by_side_comparison() function...\n")
    #plt.show()


def plot_pairwise_distances(df): #Function to plot pairwise distances for side-by-side comparison
    print("\n\tEntering plot_pairwise_distances() function...\n")
    distances = pdist(df, metric='euclidean')
    square_distances = squareform(distances)
    plt.imshow(square_distances, cmap='hot', interpolation='nearest')
    plt.colorbar()
    print("\n\tExiting  plot_pairwise_distances() function...\n")


def tsne_visualization(df, color, label): #Function to plot t-SNE visualization for side-by-side comparison
    print("\n\tEntering tsne_visualization() function...\n")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(df)
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], color=color, label=label)
    plt.legend()
    print("\n\tExiting  tsne_visualization() function...\n")


def fft_visualization(df, color, label): #Function for FFT visualization for side-by-side comparison
    print("\n\tEntering fft_visualization() function...\n")
    column_data = df.iloc[:, 0].values #FFT for the single column passed

    #Apply FFT to the NumPy array
    fft_values = fft(column_data)
    n = len(fft_values)
    freq = np.fft.fftfreq(n, d=1) #Adjust 'd' as per your time interval

    #Only take the first half of frequencies, as FFT output is symmetrical
    freq = freq[:n//2]
    fft_values = fft_values[:n//2]

    plt.plot(freq, np.abs(fft_values), color=color, label=label)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.ylim(bottom=0, top=1e5)
    plt.legend()
    print("\n\tExiting  fft_visualization() function...\n")


def describe_df(df):
    for col in [example_id_column]+convert_cols:
        if col=='datetime': continue
        temp = set()
        for i in tqdm(range(df[col].shape[0]), desc="Debug printing col="+str(col)+"...", ascii=False, ncols=75):
            temp.add(type(df[col].iloc[i]))
        print(f'{col}: nan_count={df[col].isna().sum()}; max={df[col].max()}; min={df[col].min()}; avg={df[col].mean()}; temp={temp}')
    print()


def data_type_formatting(df):
    print("Data type formatting...")
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['Sts'] = df['Sts'].astype('UInt8')
    df['example_id'] = df['example_id'].astype('UInt16')
    for feature_col in feature_cols:
        df[feature_col] = df[feature_col].astype('float64')

#def get_prominent_fft_freq(df, title, sample_pct=0.5, top_components=8):
#    freq_logging = f'\n==================== START : {title} ====================\n'
#    freq_list    = []
#
#    n_samples = int(sample_pct*len(df))
#    df_sample = df[feature_cols].head(n_samples)
#    
#    threshold = 1.5e-2 #Let's coined the term "dominant frequency" as the frequency at which dominant fft value peak is observed in FFT profile in either real DF or synthetic DF
#                       #The local neighboring frequencies surrounding the dominant frequency have approximately the same fft value as dominant peak
#                       #Since we don't want to insert many frequencies into the freq_list coming from a local neighbor of 1 particular dominant peak, we need to specify a threshold value
#                       #We only insert a particular dominant frequency candidate into freq_list if the absolute difference between it and the previously logged dominant frequency is greater than the threshold
#                       #I will explain better with each line of codes below. Please be patient with me! ;)
#
#    for col in feature_cols:
#        temp       = []
#        cold_data  = df_sample[col].values
#        fft_values = fft(col_data)
#        n          = len(fft_values)
#        freq       = np.fft.fftfreq(n, d=1)
#        fft_values = fft_values[:n//2]
#        freq       = freq[:n//2]
#
#        #temp here contains a list of pairs between fft value and its corresponding frequency
#        #List elements under temp here is sorted in descending order based on fft values
#        temp       = [ [f'{np.abs(fft_values[i]):.4f}', f'{freq[i]:.4e}'] for i in range(len(fft_values))] 
#        temp       = sorted(temp, key=lambda l:float(l[0])), reverse=True)
#
#        #temp above contains list of all fft value and frequency pairs, including those frequencies in local neighborhood of true dominant frequency
#        #temp1 below is created to only extract the true dominant frequencies out of temp
#        temp1 = []
#        for val in temp: #val here is in the form as follows [fft_value[i], freq[i]]
#            if len(temp1)==0: temp1 += [val] #Since temp is already sorted in descending order based on fft_value[i], first val must be the true dominant frequency. 
#                                             #Therefore, can just insert it directly to temp1
#
#            #do_append=False means the freq[i] is a frequency in local neighborhood of a dominant frequency
#            #do_append=True  means the freq[i] is true dominant frequency
#            do_append = True
#            for element in temp1:
#                if np.abs( float(val[1]) - float(element[1]) ) < threshold: 
#                    do_append = False #If freq[i] is close to any of previously-logged dominant frequencies, set do_append=False
#                    break
#
#            if do_append: temp1 += [val] #Inserting true dominant frequency into temp1
#
#            if len(temp1) > int(top_components) + 1: break #Once we have total count of dominant frequencies equals to top_components + 1, no need to iterate the rest of temp
#                                                           #Plus 1 here comes from the fact that frequency near 0Hz will have the largest fft value. 
#                                                           #We don't count near-0Hz frequency as dominant frequency in top_components, but we still need it to be inserted into temp1 as delta frequency calculation against threshold
#        
#        del temp #Deleting unnecessary object to prevent memory leakage during big data processing
#
#        freq_list.append(temp1) #Each temp1 corresponds to a fft_value[i] and freq[i] pair of a certain sensor column. 
#                                #So the size of 1st dim of freq_list is 3 since we have x_axis, y_axis, z_axis
#
#        freq_logging += f'\n\t{col}:\n'
#
#        for i, val in enumerate(temp1):
#            if i==0: continue
#            freq_logging += f'{val}\n'
#        freq_logging += '\n'
#
#    freq_logging = f'\n==================== FINISH: {title} ====================\n'
#
#    return freq_logging, freq_list
#
#
#def overlaid_vibration_plot(real_df, synt_df, sampleID=100):
#    #I don't have time to push these plots per sensor column into subplot of plt
#    #So, I just plot these individually in Jupyter Notebook
#    for col in feature_cols:
#        df_sample = pd.DataFrame()
#
#        df_sample['real_'+col] = real_df[real_df.example_id==sampleID][col]
#        df_sample['synt_'+col] = synt_df[synt_df.example_id==sampleID][col]
#
#        df_sample['mean_real_'+col+'_sampleID'+str(sampleID)] = real_df[real_df.example_id==sampleID][col].mean()
#        df_sample['mean_synt_'+col+'_sampleID'+str(sampleID)] = synt_df[synt_df.example_id==sampleID][col].mean()
#
#        df_sample.plot(title=f'{col}, sampleID {str(sampleID)}: Real vs Synthetic')



#CUDA-related globals
set_gpu_device(args.gpu_id)
print(f'\n\nGPU id {args.gpu_id} will be used for training...\n')

torch.cuda.empty_cache()

os.chdir(cwd_path)

if args.clean_data_from_scratch: 
    print('\n\
    #===================================================================\n\
    #=============== 01. Constructing DF in Long Format ================\n\
    #===================================================================\n')

    # Concatenating all abnormal data in a single DF (also called as Long Format) #
    abnormal_files = [f for f in os.listdir(abnormal_path_dict[failure_type]) if f.endswith('.h5')]
    i=0
    for f in abnormal_files:
        #if i>3: break
        full_path = os.path.join(abnormal_path_dict[failure_type], f)
        temp = pd.read_hdf(full_path, 'df')
        
        if temp.shape[0] > threshold_seq_len: 
            continue
        
        temp['example_id'] = i
        temp.drop(cols_to_drop, axis=1, inplace=True)
        temp['datetime'] = temp['datetime'].apply(get_date_only)
        df = pd.concat([df, temp], axis=0).reset_index(drop=True)
        max_sequence_len = max(max_sequence_len, temp.shape[0])
        print(f'Iteration ({i+1}/{len(abnormal_files)}) => temp: {temp.shape}, df: {df.shape}')
        i+=1
    print(f'df.head(5): \n{df.head(5)}\n\ndf.tail(5): \n{df.tail(5)}\n{df.dtypes}\n\n')


if args.clean_data_from_scratch:
    print('\n\
    #===================================================================\n\
    #=============== 02. Scaling between [-1, 1]        ================\n\
    #===================================================================\n')

    df[convert_cols] = df[convert_cols].apply(pd.to_numeric, errors='coerce')
    print(f'df.head(5): \n{df.head(5)}\n\ndf.tail(5): \n{df.tail(5)}\n{df.dtypes}\n\n')

    ##This minmax_offset file is used to stretch back synthetic_df to get a signal the more resemble raw data 
    ax_min_max = []
    ax_min_max.append([df[ax].min() for ax in feature_cols])
    ax_min_max.append([df[ax].max() for ax in feature_cols])
    df_minmax = pd.DataFrame(ax_min_max, columns=['x_offset', 'y_offset', 'z_offset']) #3 cols for x, y, z axis. 2 Rows: 1st row for min value, & 2nd row for max value
    df_minmax.to_csv(target_minmax_offset_filepath)
    print(f'Saving minmax_offset file post standardizing is complete...\n')

    ## Standardization N(0, 1)
    #for ax in feature_cols:
    #    df[ax] = (df[ax]-df[ax].mean())/df[ax].std()
    #print(f'x_axis: {df.x_axis.mean():.2f}, {df.x_axis.std():.2f} | y_axis: {df.y_axis.mean():.2f}, {df.y_axis.std():.2f} | z_axis: {df.z_axis.mean():.2f}, {df.z_axis.std():.2f\n}')

    ## Scaling between [-1, 1]
    for ax in feature_cols:
        df[ax] = (df[ax] - df[ax].min())/(df[ax].max()-df[ax].min())*2 - 1

    df.to_hdf(target_h5_real_filepath, key='df', mode='w')
    print(f'Saving h5real file at the end of Section 02 is complete...\n')


if args.clean_data_from_scratch or args.data_imputation:
    print('\n\
    #===================================================================\n\
    #=== 03. Data Imputation by Replication of Last 1000 Timestamps ====\n\
    #===================================================================\n')

    df = pd.read_hdf(target_h5_real_filepath, 'df').reset_index(drop=True)
    print(f'\n\nRight after reading h5 file:\ndf.head(5):\n{df.head(5)}\n\ndf.tail(5):\n{df.tail(5)}\n{df.dtypes}\n\n')
    #describe_df(df)
    print(f'Reading h5_real DataFrame at the beginning of Section 03 is complete...\n')

    max_sequence_len, df = populate_each_sampleID(df, num_imputation_repeat, example_id_column)

    grouped_sizes = df.groupby(example_id_column).size()
    if grouped_sizes.unique()[0] != max_sequence_len: print(f'\nMismatch detected: Expected max_sequence_len={max_sequence_len}, but found {grouped_sizes.unique()[0]}')
    else: print(f'\nNo mismatch detected: sequence length of each example_id is the same as max_sequence_len={max_sequence_len}')

    df.to_hdf(target_h5_impt_filepath, key='df', mode='w')
    print(f'Saving output h5 file at the end of Section 03 is complete...\n')


if args.train:
    print('\n\
    #===================================================================\n\
    #=============== 04. Training using DoppelGANger    ================\n\
    #===================================================================\n')

    df = pd.read_hdf(target_h5_impt_filepath, 'df').reset_index(drop=True)
    max_sequence_len = df.groupby(example_id_column).size().unique()[0]
    data_type_formatting(df)
    df.drop(['datetime'], axis=1, inplace=True)
    print(f'\n\ndf.head(5):\n{df.head(5)}\n\ndf.tail(5):\n{df.tail(5)}\n{df.dtypes}\n\nmax_sequence_len = {max_sequence_len}\n\n')
    print(f'Reading h5_real DataFrame at the beginning of Section 04 is complete...\n')

    model=DGAN(DGANConfig(max_sequence_len                     =max_sequence_len,   
                          sample_len                           = args.sample_len, 
                          batch_size                           = args.batch_size, 
                          epochs                               = args.epochs, 
                          attribute_num_layers                 = args.attribute_num_layers, 
                          attribute_num_units                  = args.attribute_num_units, 
                          feature_num_layers                   = args.feature_num_layers, 
                          feature_num_units                    = args.feature_num_units, 
                          generator_learning_rate              = args.generator_learning_rate, 
                          discriminator_learning_rate          = args.discriminator_learning_rate, 
                          attribute_discriminator_learning_rate= args.attribute_discriminator_learning_rate, 
                          apply_feature_scaling                = args.apply_feature_scaling, 
                          discriminator_rounds                 = args.discriminator_rounds,
                          generator_rounds                     = args.generator_rounds,
                          normalization  = Normalization.MINUSONE_ONE, #Default normalization is ZERO_ONE. Since we do standardize the data into N(0, 1), it should be updated to MINUSONE_ONE 
                         ),
               cwd_path                 = cwd_path,
               exp_id                   = args.exp_id,
               epoch_patience           = args.epoch_patience,
               delta_loss_patience      = args.delta_loss_patience,
               num_epoch_per_checkpoint = args.num_epoch_per_checkpoint,
               oscillate_DLR            = args.oscillate_DLR,
               oscillate_GLR            = args.oscillate_GLR,
               KLdiv_loss_replacement   = args.KLdiv_loss_replacement,
               KLdiv_loss_addition      = args.KLdiv_loss_replacement,
               KLdiv_fft_loss_addition  = args.KLdiv_fft_loss_addition,
               spectral_loss_addition   = args.spectral_loss_addition,
              )  
    print(f'Setting up DGANConfig is complete...\n')

    #Setting up TensorBoard
    writer = SummaryWriter(log_dir=f'{target_tensorboard_filepath}')

    #Dropping attribute_columns since the values of this attribute_columns must be constant under the same example_id_column. 
    #  DGAN will be trained with attributes=None
    #In view of this, column 'Sts' will be included under feature_columns. 
    #  Please note that convert_cols below is equivalent to 'Sts' + feature_cols
    #Dropping discrete_columns since we have no column with string dtype
    model.train_dataframe(df,
                          #attribute_columns=['Sts'],          #Columns that do not vary across timestamps
                          feature_columns=convert_cols,        #Columns that        vary across timestamps
                          example_id_column=example_id_column, #A column that indicates the sample ID number
                          #time_column="datetime",             #A column that comprises information of date only
                          #discrete_columns=['Sts'],           #Columns that are of discrete in nature
                          df_style="long",                     #"long" for multi-variate time-series dataset and "wide" for univariate time-series dataset
                          progress_callback=writer
                         )
    print(f'Training model is complete...\n')

    model.save(target_model_filepath)
    print(f'Saving trained model is complete...\n')


if args.train or args.generate_synthetic:
    print('\n\
    #===================================================================\n\
    #=============== 05. Generating Synthetic Data      ================\n\
    #===================================================================\n')

    df = pd.read_hdf(target_h5_impt_filepath, 'df').reset_index(drop=True)
    print(f'\n\ndf.head(5):\n{df.head(5)}\n\ndf.tail(5):\n{df.tail(5)}\n\n')
    print(f'Reading h5_real DataFrame at the beginning of Section 05 is complete...\n')

    model = DGAN.load(target_model_filepath)
    print(f'Loading saved model is complete...\n')

    synthetic_df = model.generate_dataframe(409)
    synthetic_df.to_hdf(target_h5_synt_filepath, 'df')
    print(f'\n\nsynt_df.head(5):\n{synthetic_df.head(5)}\n\nsynt_df.tail(5):\n{synthetic_df.tail(5)}\n\n')
    print(f'Saving output h5 file at the end of Section 04 is complete...\n')


if args.train or args.generate_synthetic or args.evaluate:
    print('\n\
    #===================================================================\n\
    #=============== 06. Evaluating Synthetic Data      ================\n\
    #===================================================================\n')

    #  Phase 01: - To keep this performance evaluation section at the end of the script once all training is completed
    #            - This is done to ensure we can get exact gauge of how long the training is gonna take
    #            - Further, we need to ensure smoothness of the training as top priority first. Once we settle training portion, we can interlace training performance evaluation per training epoch


    df = pd.read_hdf(target_h5_impt_filepath, 'df').reset_index(drop=True)
    print(f'\n\ndf.head(5):\n{df.head(5)}\n\ndf.tail(5):\n{df.tail(5)}\n\n')
    print(f'Reading h5_real DataFrame is complete...\n')

    synthetic_df = pd.read_hdf(target_h5_synt_filepath, 'df')
    print(f'\n\nsynt_df.head(5):\n{synthetic_df.head(5)}\n\nsynt_df.tail(5):\n{synthetic_df.tail(5)}\n\n')
    print(f'Reading h5_synt DataFrame is complete...\n')

    #Plots
    std_dev = 0.3  # Amount of gaussian noise to inject
    gauss_df = add_gaussian_noise(df, std_dev, feature_cols)
    extended_plot_distributions(df, synthetic_df, gauss_df, "KDE")
    side_by_side_comparison(df, synthetic_df, gauss_df, fft_visualization, "FFT") 
    #overlaid_vibration_plot(df, synthetic_df)

    #Quality Metrics Printing
    metrics = data_qm.evaluate_synthetic_data_multivariate(df[feature_cols], synthetic_df[feature_cols], feature_cols, max_sequence_len)
    with open(target_metric_filepath, "w") as file1:
        file1.write(data_qm.print_metrics_to_file(metrics, feature_cols))

        ##For logging on FFT frequencies with dominant peaks for both df and synthetic_df
        #real_df_freq_logging, real_df_freq_list = get_prominent_fft_freq(          df, 'real_df')
        #synt_df_freq_logging, synt_df_freq_list = get_prominent_fft_freq(synthetic_df, 'synt_df')
        #file1.write(real_df_freq_logging)
        #file1.write(synt_df_freq_logging)


    #  Phase 02: - To interlace training performance evaluation per training epoch.
    #             - In order to do this, we need to copy over the source code from gretel-synthetics github repo and add few lines to record the performance metrics into tensorboard

    # Not done yet
