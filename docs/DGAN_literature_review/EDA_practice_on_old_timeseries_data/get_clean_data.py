import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import matplotlib.pyplot as plt
import networkx as nx
import os 
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
import numpy as np
import h5py
import time
from tqdm import tqdm
from collections import defaultdict

from sklearn.cluster import OPTICS, Birch, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, minmax_scale

import seaborn as sns

import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


train=1
get_data=0


def data_processing(master_file, ref_axis, filename, skip=0, normalize=1):
    #skip = 0 --> Extract data, Save data, Read data, Euclidean data
    #skip = 1 --> Read data, Euclidean data
    
    def get_machine(x):
        name = x.split("_")
        return name[0]

    def get_process(x):
        name = x.split("_")
        return name[3]
    
    #----------------------------------------------------------------------------------------------------------------------------
    if skip==0:
        # Consolidate all filepaths to each dataset
        filepaths = []
        with open(master_file) as files:
            filepaths = files.read().split('\n')
        numfiles_to_parse = int(len(filepaths))-1 #Need to substract by one since the last element is an empty path 
        final_df = pd.DataFrame()


        # Consolidate Normalized Data into DataFrame
        itr = 0
        for numfile in tqdm(range(numfiles_to_parse), desc=axis_name+": Consolidate Normalized Data into DataFrame…", ascii=False, ncols=75):
            path = filepaths[numfile]
            df = pd.DataFrame(h5py.File(filepaths[numfile],'r')['vibration_data'])
            df.columns = ['xaxis', 'yaxis','zaxis']
            aux = pd.DataFrame(df.values.T)
            aux['axis'] = [str(col) for col in df.columns]
            aux['label']= 'good' if 'good' in filepaths[numfile] else 'bad' 
            aux.index = [filepaths[numfile].split('/')[-1][:-3]+'_'+filepaths[numfile].split('/')[-2]]*3
            scaled_matrix = minmax_scale(aux[aux.columns[:-2]].T).T if normalize else aux[aux.columns[:-2]]
            df = pd.DataFrame(scaled_matrix)
            df.index = aux.index
            df['axis'] = aux['axis']
            df['label']= aux['label']
            df = df[df.axis==ref_axis]
            final_df = pd.concat([final_df, df], axis=0)
            del df, aux
            #print(f'Data Processing (Consolidation): {ref_axis} => {itr}/{numfiles_to_parse}')
            itr+=1

        # Place columns 'axis' and 'label' as first 2 columns
        aux = pd.DataFrame()
        aux = pd.concat([final_df['axis'],final_df['label'],final_df.loc[:,0:268287],final_df.loc[:,268288:]],axis=1)
        final_df = pd.DataFrame()
        final_df = aux
        #final_df.to_hdf('final_df.h5', key='stage', mode='w')

        # Replicate last 1000 data points
        numrows = final_df.shape[0]
        num_replicate = 1000
        itr=0
        for row in tqdm(range(numrows), desc=axis_name+": Replicate last 1000 data points…", ascii=False, ncols=75):
            temp = final_df.iloc[row, 2:]
            nan_idx = -1
            for idx, val in enumerate(temp):
                if np.isnan(val):
                    nan_idx = idx
                    break
            if nan_idx!=-1:
                for idx in range(nan_idx,int(len(temp))):
                    temp[idx] = temp[idx-1000]
            final_df.iloc[row, 2:] = temp
            #print(f'Data Processing (Replication): {ref_axis} => {itr}/{numrows}')
            itr+=1

        # Granularize by Process
        aux = pd.DataFrame()
        aux['Machine'] = final_df.index
        aux['Machine'] = aux['Machine'].apply(get_machine)
        aux['Process'] = final_df.index
        aux['Process'] = aux['Process'].apply(get_process)
        final_df.insert(loc=0, column='Machine', value=aux.Machine.to_list())
        final_df.insert(loc=1, column='Process', value=aux.Process.to_list())
            
        # Save file
        final_df.to_hdf(filename, key='stage', mode='w')
        del aux
    #----------------------------------------------------------------------------------------------------------------------------
    
    # Read file
    final_df   = pd.read_hdf(filename,'stage')
    
    return final_df



def get_euclidean_data(final_df):
    # Calculate Euclidean Data
    X = final_df.drop('axis', axis=1).drop('label',axis=1).drop('Machine',axis=1).drop('Process',axis=1).values
    dist = euclidean_distances(X, X)
    dist = pd.DataFrame(dist)
    dist.index = final_df.index
    dist.columns = dist.index
    
    return dist

    
        
def tsne_func(final_df, dist, axis_name, pc_count=2, perplexity=10):
    X_embedded = pd.DataFrame(TSNE(n_components=pc_count, learning_rate='auto',
                      init='random', perplexity=perplexity).fit_transform(dist.values))
    X_embedded.index = dist.index
    X_embedded.columns = ["pc1", "pc2"] if pc_count==2 else ["pc1", "pc2", "pc3"]

    cmap = []
    for val in final_df.label.to_list(): 
        cmap += [1] if val=='good' else [0]

    if pc_count==2:
        row, col, plot_inc = 2, 2, 1
        fig = plt.figure(figsize=(7,7))
        
        #'Actual Label':
        plt.subplot(row, col, plot_inc)
        plt.scatter(X_embedded.loc[:,"pc1"], X_embedded.loc[:,"pc2"], c=cmap)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(axis_name + ": Actual Label")
        plot_inc+=1
        
        clustering      = [Birch(          n_clusters=2).fit(X_embedded.values), 
                           OPTICS(       min_samples=10).fit(X_embedded.values), 
                           DBSCAN(eps=3, min_samples=10).fit(X_embedded.values)]
        clustering_name = ["Birch",
                           "OPTICS",
                           "DBSCAN"]

        for i, model in enumerate(clustering):
            plt.subplot(row, col, plot_inc)
            plt.scatter(X_embedded["pc1"], X_embedded["pc2"], c=model.labels_)
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title(axis_name + ": " + clustering_name[i]+" Label")
            plot_inc+=1        

    if pc_count==3:
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(projection='3d')
        ax.scatter(X_embedded.loc[:,"pc1"], X_embedded.loc[:,"pc2"], X_embedded.loc[:,"pc3"], c=cmap)
        ax.set_zlabel("Principal Component 3")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2") 
        plt.title(axis_name + ": Actual Label")
        
    fig.tight_layout()
    plt.show()
    
def onehot_encoding(x):
    return [1, 0] if x=='good' else [0, 1]


def dataset_split(dataset, good_idx, bad_idx):
    ans = []
    for count in range(int(len(good_idx))):
        ans += [dataset[good_idx[count]]]
    for count in range(int(len(bad_idx))):
        ans += [dataset[bad_idx[count]]]
    return ans







if get_data:
    #===================================================================================================================================
    #Create 3 post-processed files: final_df_xaxis_process.h5, final_df_yaxis_process.h5, final_df_zaxis_process.h5"
    #===================================================================================================================================
    master_file = "./file_list.log"    
    axis_names  = ['xaxis', 'yaxis', 'zaxis']

    for axis_name in axis_names:
        print("====================================================================================")
        filename    = "final_df_"+axis_name+"_process.h5"
        final_df = data_processing(master_file, axis_name, filename, skip=0)
    print("Done creating 3 post-processed files")

    #===================================================================================================================================
    #Create 1 consolidated Step_Drill file: input_df_Step_Drill.h5
    #===================================================================================================================================
    drill_dict = {'Step_Drill'    : ['OP00', 'OP01', 'OP03', 'OP04', 'OP05', 'OP06', 'OP07', 'OP08', 'OP10', 'OP11', 'OP12', 'OP14'],
                  'Drill'         : ['OP02'],
                  'T-Slot_Cutter' : ['OP13'],
                  'Straight_Flute': ['OP09']
                 }
    axis_names   = ['xaxis', 'yaxis', 'zaxis']
    ref_drill_type = "Step_Drill"

    input_df = pd.DataFrame()

    for axis in axis_names:
        print(f'{axis}: Start...')
        temp = pd.read_hdf("final_df_"+axis+"_process.h5",'stage')
        temp = temp[temp.Process.isin(drill_dict[ref_drill_type])]
        print(f'{axis}: Done Reading...')
        input_df = pd.concat([input_df, temp], axis=0)
        print(f'{axis}: Done Concatenating...')
        del temp
  
    input_df.to_hdf("input_df_Step_Drill.h5", key='stage', mode='w')
    print('Done saving input_df_Step_Drill.h5')







if train:
    #===================================================================================================================================
    #Anomaly Detection with XGBOOST
    #===================================================================================================================================
    input_df   = pd.read_hdf("input_df_Step_Drill.h5",'stage')
    input_df['label'] = input_df.label.apply(onehot_encoding)
    print('Done performing onehot_encoding transformation on input_df[\'label\']')
    
    x_dataset = input_df.iloc[:, 4:].to_numpy()
    y_dataset = input_df['label'].to_numpy()
    print(f'Done gathering x_dataset ({x_dataset.shape}) and y_dataset ({y_dataset.shape})')
    
    
    train_ratio = 0.7
    val_ratio   = 0.2
    test_ratio  = 0.1
    
    good_idx, bad_idx = [], []
    for idx, label in enumerate(y_dataset):
        if label == [1,0]: good_idx += [idx]
        if label == [0,1]:  bad_idx += [idx]
    
    good_count, bad_count = int(len(good_idx)), int(len(bad_idx))
    
    train_good, train_bad = int(train_ratio*good_count), int(train_ratio*bad_count)
    val_good,     val_bad = int(  val_ratio*good_count), int(  val_ratio*bad_count)
    test_good, test_bad = good_count-train_good-val_good, bad_count-train_bad-val_bad
    print("\nDone counting good data and bad data...")
    print(f'train_good={train_good}, train_bad={train_bad}')
    print(f'val_good  ={val_good},   val_bad  ={val_bad}')
    print(f'test_good ={test_good},  test_bad ={test_bad}')
    
    
    x_train = np.asarray(dataset_split(x_dataset,                    good_idx[:train_good],                  bad_idx[:train_bad]))
    x_val   = np.asarray(dataset_split(x_dataset, good_idx[train_good:train_good+val_good], bad_idx[train_bad:train_bad+val_bad]))
    x_test  = np.asarray(dataset_split(x_dataset,           good_idx[train_good+val_good:],          bad_idx[train_bad+val_bad:]))
    
    y_train = np.asarray(dataset_split(y_dataset,                    good_idx[:train_good],                  bad_idx[:train_bad]))
    y_val   = np.asarray(dataset_split(y_dataset, good_idx[train_good:train_good+val_good], bad_idx[train_bad:train_bad+val_bad]))
    y_test  = np.asarray(dataset_split(y_dataset,           good_idx[train_good+val_good:],          bad_idx[train_bad+val_bad:]))
    
    print(f'\nDone splitting training, validation, and test datasets')
    print(f'x_train ({x_train.shape}), y_train ({y_train.shape})')
    print(f'x_val   ({  x_val.shape}), y_val   ( { y_val.shape})')
    print(f'x_test  ({ x_test.shape}), y_test  ( {y_test.shape})')
