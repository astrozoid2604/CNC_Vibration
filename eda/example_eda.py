import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os, re, math, h5py
import utils.utility as utility
import seaborn as sns
from parallel_pandas import ParallelPandas

def main():
    #initialize parallel-pandas
    ParallelPandas.initialize(n_cpu=16, split_factor=4, disable_pr_bar=False)

    en_store_h5_format = False
    en_load_h5_dirs = False
    machine_type = 'AHC-A3'
    en_EDA = True
    en_feature_eng = False
    en_read_single_file = False

    if (en_EDA):
        n_normal = 90 # num of normal samples to read
        n_abnormal = n_normal # num of normal samples to read    
    else:
        n_normal = 100 # num of normal samples to read
        n_abnormal = int(0.2*n_normal) # fraction of normal samples to read

    # Starting root folder where Machine IDs are located
    root_path = os.getcwd() + '/data/' + machine_type
    curr_path = os.getcwd()
    # print(root_path)

    save_path='plots'
    
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if (en_store_h5_format):
        utility.process_files_test(root_path)
    
    if (en_load_h5_dirs):
        machine_data = utility.load_h5_files_test2(root_path)
        
        # Combine list of DataFrames into a single DataFrame
        combined_df = pd.concat(machine_data, ignore_index=True)
        print(combined_df.head(20))
        print(combined_df.info())

    if (en_read_single_file):
        '''
        Ball-Screws
        '''
        # Abnormal files
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Ball_screw\Ball_screw_220916\\2022-09-16-04-37-37-91.h5' #Eg 1 
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Ball_screw\Ball_screw_220916\\2022-09-16-04-42-38-43.h5' # Eg 2 
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Ball_screw\Ball_screw_220916\\2022-09-16-07-42-53-30.h5' # Eg 3
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Ball_screw\Ball_screw_220916\\2022-09-16-09-11-61-23.h5' #Eg 4 
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Ball_screw\Ball_screw_220916\\2022-09-16-08-45-59-45.h5' #Eg 5 
        # ----------------------

        '''
        BartoBarFailure
        '''
        # Abnormal
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\BartoBarFailure\BartoBarFailure_20220328\\2022-03-28-10-29-00-03.h5' #Eg 1
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\BartoBarFailure\BartoBarFailure_20220328\\2022-03-28-10-31-00-17.h5' #Eg 1
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\BartoBarFailure\BartoBarFailure_20220328\\2022-03-28-10-33-00-33.h5' #Eg 1
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\BartoBarFailure\BartoBarFailure_20220328\\2022-03-28-11-00-00-55.h5' #Eg 1
        # abnormal_file_path = curr_path + '\\data\\data\AHC-A3\\abnormal\BartoBarFailure\BartoBarFailure_20220328\\2022-03-28-11-01-00-62.h5' #Eg 1
        # ----------------------

        '''
        Belt Wheel
        '''
        # Abnormal
        abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Belt_wheel\Belt wheel_220228\\Product_96.h5' #Eg 1
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Belt_wheel\Belt wheel_220228\\Product_82.h5' #Eg 1
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Belt_wheel\Belt wheel_220228\\Product_7.h5' #Eg 3
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Belt_wheel\Belt wheel_220228\\1_Product_75.h5' #Eg 4
        # abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Belt_wheel\Belt wheel_220228\\1_Product_3.h5' #Eg 5
        # ----------------------

        # Normal files
        normal_file_path = curr_path + '\\data\AHC-A3\\normal\\normal_210908\\1_Product_1.h5' # Eg 1
        # normal_file_path = curr_path + '\\data\AHC-A3\\normal\\normal_220228\\1_Product_1.h5' # Eg 2
        # normal_file_path = curr_path + '\\data\AHC-A3\\normal\\normal_220302\\1_Product_1.h5' # Eg 3
        # normal_file_path = curr_path + '\\data\AHC-A3\\normal\\normal_221008\\2022-10-08-19-11-44-26.h5' # Eg 4
        # normal_file_path = curr_path + '\\data\AHC-A3\\normal\\normal_220302\\1_Product_30.h5' # Eg 5

        normal_dfs, abnormal_dfs = utility.read_hdf_files(normal_file_path, abnormal_file_path)
    else:
        abnormal_file_path = curr_path + '\\data\AHC-A3\\abnormal\Ball_screw\Ball_screw_220916'
        normal_file_path = curr_path + '\\data\AHC-A3\\normal\\normal_210908'
        
        # Data Import via random file reading
        print('Importing in progress...')
        normal_dfs, abnormal_dfs = utility.read_hdf_files_random_test(normal_file_path, 
                                                                        abnormal_file_path, 
                                                                        n_normal, 
                                                                        n_abnormal)
        print("Completed - Data Importing \n")
    
    # Date Pre-processing
    # -----------------
    # Convert object columns to float64 - Some files have inconsistent formats. No idea why.
    print('Data Preprocessing in progress')
    convert_cols = ['Sts', 'x_axis', 'y_axis', 'z_axis']
    cols_to_drop = ['date', 'time']

    if (en_read_single_file):
        abnormal_dfs[convert_cols] = abnormal_dfs[convert_cols].apply(pd.to_numeric, errors='coerce')
        normal_dfs[convert_cols] = normal_dfs[convert_cols].apply(pd.to_numeric, errors='coerce')

        normal_dfs = normal_dfs.drop(cols_to_drop, axis=1)
        abnormal_dfs = abnormal_dfs.drop(cols_to_drop, axis=1)
    else:
        # Apply the same preprocessing to all DataFrames in the dictionaries
        for file, df in normal_dfs.items():
            # print(f'Processing Normal class filename: {file}')
            df[convert_cols] = df[convert_cols].apply(pd.to_numeric, errors='coerce')
            df.drop(cols_to_drop, axis=1, inplace=True)
            df.set_index('datetime', inplace=True)

        for file, df in abnormal_dfs.items():
            # print(f'Processing Abormal class filename: {file}')
            df[convert_cols] = df[convert_cols].apply(pd.to_numeric, errors='coerce')
            df.drop(cols_to_drop, axis=1, inplace=True)
            df.set_index('datetime', inplace=True)

    print("Completed - Data pre-processing \n")

    # print(normal_df.columns.values)
    # print(abnormal_df.columns.values)
    # print(normal_df.head())
    # print(abnormal_df.head())
    # print(type(normal_dfs))

    # TODO: EDA of Vibration data
    # --------------
    if (en_EDA):
        print('EDA enabled')

        # Example usage:
        # utility.create_save_pairplot(normal_dfs, 'Pairplot of Normal Vibration Data', 'normal_vibration_data')
        # utility.create_save_pairplot(abnormal_dfs, 'Pairplot of Abnormal Vibration Data', 'abnormal_vibration_data')
        utility.create_save_combined_pairplot_test(normal_dfs, abnormal_dfs, 'vibration_data', verbose=True)
        # (Optional) Debugging data types
        # print("Data types in normal_df:")
        # print(normal_df.dtypes)
        # print("\nData types in abnormal_df:")
        # print(abnormal_df.dtypes)
        # utility.plot_side_by_side_plotly(normal_df, abnormal_df)

        # # Plot Frequency Analysis results
        # utility.plot_vibration_fft_and_stft(normal_df, abnormal_df)
    # --------------

    # Data Pre-processing
    # --------------
    # Assuming normal_dfs and abnormal_dfs are the dictionaries of DataFrames you got from read_hdf_files_random
    print('Data Preprocessing initiated')
    preprocessed_normal, max_length_normal = utility.preprocess_data_with_padding(normal_dfs)
    preprocessed_abnormal, max_length_abnormal = utility.preprocess_data_with_padding(abnormal_dfs, max_length=max_length_normal)

    print(f'Max length of normal is: {max_length_normal}')
    print(f'Max length of abnormal is: {max_length_abnormal}')
    # print(preprocessed_normal[:3])  # This will display the first 3 elements of the list
    # print(preprocessed_abnormal[:3])  # This will display the first 3 elements of the list
    print('Completed - Data Preprocessing \n')
    # --------------


if __name__ == "__main__":
    main()
    print('Application completed. Exiting now.')
