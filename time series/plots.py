# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:28:02 2022

@author: agnie
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:34:21 2022

@author: agnie
"""
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import statistics as st
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller

from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad



SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\wyniki"





# Function to read file
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
                  
    return dataset

#Function to get number of files with specific extension:
def counter_files(path, extension):
    list_dir = []    
    list_dir = os.listdir(path)
    count = 0
    for file in list_dir:
        if file.endswith(extension):
            count+=1
    return  count


#Function to get list of files with specific extension:
def get_list_files(path,extension):
    directory = os.fsencode(path)
    list_files = []
    only_files_name = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(extension):
           list_files.append(path + '/' +filename)
           only_files_name.append(filename)
    return list_files, only_files_name


#function to find extreme values
def outliers(data, ex):
    total_cols=len(data.axes[1])
    
    for i in range (0, total_cols):
        kolumna = data.iloc[:,i]
                   
        q1 = kolumna.quantile(q =0.25)
        q3 = kolumna.quantile(q = 0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-ex*iqr
        fence_high = q3+ex*iqr
        df_out = kolumna.loc[(kolumna < fence_low )|(kolumna > fence_high)]
        kolumna[df_out.index] = None
        
    return data

def interpolate_gaps(values, limit=None):
    """
    Fill gaps using linear interpolation, optionally only fill gaps up to a
    size of `limit`.
    """
    values = np.asarray(values)
    i = np.arange(values.size)
    valid = np.isfinite(values)
    filled = np.interp(i, i[valid], values[valid])

    if limit is not None:
        invalid = ~valid
        for n in range(1, limit+1):
            invalid[:-n] &= invalid[n:]
        filled[invalid] = np.nan

    return filled



def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
   """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

def rolling_spearman(seqa, seqb, window):
    stridea = seqa.strides[0]
    ssa = as_strided(seqa, shape=[len(seqa) - window + 1, window], strides=[stridea, stridea])
    strideb = seqa.strides[0]
    ssb = as_strided(seqb, shape=[len(seqb) - window + 1, window], strides =[strideb, strideb])
    ar = pd.DataFrame(ssa)
    br = pd.DataFrame(ssb)
    ar = ar.rank(1)
    br = br.rank(1)
    corrs = ar.corrwith(br, 1)
    return pad(corrs, (window - 1, 0), 'constant', constant_values=np.nan)

#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)



results_iteration = []
for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ',', decimal_sign = '.')
    dataset.drop(['DateTime','ABP'], axis=1, inplace=True)
    dataset = dataset.rename(columns={'brs': 'BRS'})
    
    ################graph to show raw data
    
    dataset_raw = read_file(files_list[counter_files], separator = ',', decimal_sign = '.')
    dataset_raw = dataset_raw.rename(columns={'brs': 'BRS'})
    
    f,(ax1,ax2)=plt.subplots(2,1,figsize=(10,5),sharex='col')
    ax1.plot(dataset_raw['BRS'])
    ax2.plot(dataset_raw['ICP'])
    ax1.set(ylabel='ICP [mm Hg]')
    ax2.set(ylabel = 'BRS [ms/mm Hg]')
    ax1.set(title=f"Raw signal BRS and ICP")
    ax2.set(xlabel = 'Time [Hours]')
    #points = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000 ]
    #points2 = []
    #for i in range(0,len(points)):
        #points2.append(points[i]/360)
    #points2 = [round(elem, 2) for elem in points2 ]
    #ax2.set_xticks(points)
    #ax2.set_xticklabels(points2)
    
    ####################outliers -out
    dataset=outliers(dataset,3)
    
    ###########linear interpolation
    filled_i = interpolate_gaps(dataset.iloc[:,0], limit=6)
    filled_b = interpolate_gaps(dataset.iloc[:,1], limit=6)
    dataset_i = pd.DataFrame(filled_i)
    dataset_b = pd.DataFrame(filled_b)
    frames = [dataset_i, dataset_b]
    dataset = pd.concat(frames,axis=1)
    dataset.columns = ['ICP', 'BRS']
    
    ###############graph to show data without ouliers and interpolate
    f,(ax1,ax2)=plt.subplots(2,1,figsize=(10,5),sharex='col')
    ax1.plot(dataset['BRS'])
    ax2.plot(dataset['ICP'])
    ax1.set(ylabel='BRS[mm Hg]')
    ax2.set(ylabel = 'ICP [ms/mm Hg]')
    ax1.set(title=f"Raw signal BRS and ICP")
    ax2.set(xlabel = 'Time [Hours]')
    #points = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000 ]
    #points2 = []
    #for i in range(0,len(points)):
        #points2.append(points[i]/360)
    #points2 = [round(elem, 2) for elem in points2 ]
    #ax2.set_xticks(points)
    #ax2.set_xticklabels(points2)
    
    overall_pearson_r = dataset['BRS'].corr(dataset['ICP'])
    overall_spearman_r = dataset['BRS'].corr(dataset['ICP'], method = 'spearman')
    
    
    ############median rolling filter -only to show -applied to graph
    f,ax=plt.subplots(figsize=(10,5))
    dataset.rolling(window=720,center=True,min_periods=1).median().plot(ax=ax)
    ax.set(xlabel='Time [Hours]',ylabel='mm Hg or ms/mm Hg',xlim=[0,35000])
    points = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000 ]
    points2 = []
    for i in range(0,len(points)):
        points2.append(points[i]/360)
    points2 = [round(elem, 2) for elem in points2 ]
    ax.set_xticks(points)
    ax.set_xticklabels(points2)
    ax.set(title=f"Pearson r: {overall_pearson_r:.2f}\n\
    Spearman r: {overall_spearman_r:.2f} ")
    
    
    ################################
    # Set window size to compute moving window synchrony.
    r_window_size = 360 #######1 hour
    
    # Compute rolling window synchrony
    rolling_r = dataset['BRS'].rolling(window=r_window_size, center=True, min_periods=1).corr(dataset['ICP'])
    f,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    dataset.rolling(window=360,center=True,min_periods=1).median().plot(ax=ax[0])
    ax[0].set(xlabel='Time [Hours]',ylabel='mm Hg or ms/mm Hg')
    rolling_r.plot(ax=ax[1])
    
    #points = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000 ]
    #points2 = []
    #for i in range(0,len(points)):
        #points2.append(points[i]/360)
    #points2 = [round(elem, 2) for elem in points2 ]
    #ax2.set_xticks(points)
    #ax2.set_xticklabels(points2)
    ax[0].set(title="ICP, BRS data and rolling window correlation\n\
    time window = 2 hours")
    ax[1].set(label='Time [Hours]',ylabel='Perason r')
    
    #############rolling Spearman
    w = rolling_spearman(dataset.BRS.to_numpy(), dataset.ICP.to_numpy(), 720)
    
    f,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    dataset.rolling(window=360,center=True,min_periods=1).median().plot(ax=ax[0])
    ax[0].set(xlabel='Time [Hours]',ylabel='mm Hg or ms/mm Hg')
    plt.plot(w)
    
    #points = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000 ]
    #points2 = []
    #for i in range(0,len(points)):
        #points2.append(points[i]/360)
    #points2 = [round(elem, 2) for elem in points2 ]
    #ax2.set_xticks(points)
    #ax2.set_xticklabels(points2)
    ax[0].set(title="ICP, BRS data and rolling window correlation\n\
    time window = 2 hours")
    ax[1].set(label='Time [Hours]',ylabel='Spearman r')
    
    #########TLCC#############
    
    rs = [crosscorr(dataset.BRS,dataset.ICP, lag) for lag in range(-180,180+1)]
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    
    f,ax=plt.subplots(figsize=(10,5))
    ax.plot(rs)
    ax.axvline(np.floor(len(rs)/2),color='k',linestyle='--',label='Center')
    ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
    ax.set(title=f'Offset = {offset/6:.2f} minutes\n BRS vs. ICP', xlim=[0,361],xlabel='Offset',ylabel='Pearson r')
    points = [0, 50, 100, 150, 200, 250, 300, 350]
    ax.set_xticks(points)
    ax.set_xticklabels([-175, -125, -75,-25, 25, 75, 125, 175])
    plt.legend()
    
     ################ Windowed time lagged cross correlation

    
    no_splits = 20
    samples_per_split = dataset.shape[0]/no_splits
    rss=[]
    
    for t in range(0, no_splits):
        d1 = dataset['BRS'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        d2 = dataset['ICP'].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(180),int(180+1))]
        rss.append(rs)
    
    rss = pd.DataFrame(rss)
    

    f,ax = plt.subplots(figsize=(10,5))
    sns.heatmap(rss,cmap='RdBu_r',ax=ax)
    ax.set(title=f'Windowed Time Lagged Cross Correlation',xlim=[0,361], xlabel='Offset',ylabel='Window epochs')
    points = [0, 50, 100, 150, 200, 250, 300, 350]
    ax.set_xticks(points)
    ax.set_xticklabels([-175, -125, -75,-25, 25, 75, 125, 175]);
    
    