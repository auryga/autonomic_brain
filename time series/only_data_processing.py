import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import statistics as st
import matplotlib.pyplot as plt


import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from numpy.lib.stride_tricks import as_strided
from numpy.lib import pad

from numpy import ndarray



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

#function to interpolate data
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


#function to cross-correlate data
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


def xycorr_df (x,y):
    a = x.to_numpy()
    b = y.to_numpy()
    
    a = a.conj().T
    b = b.conj().T
    
    len_a = len(a)
    len_b = len(b)
    
    c = np.array([np.zeros(len_a), a])
    d = np.array([np.zeros(len_b), b])
    
#######Compute FFTs
    X1 = np.fft.fft(c)
    X2 = np.fft.fft(d)
    
##########3Compute cross correlation 
    X = X1*np.conj(X2)
    ck = np.fft.ifft((X))
    
    td  = np.argmax(ck) - len_a
    
    return td
















#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)



results_iteration = []
for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ',', decimal_sign = '.')
    dataset.drop(['DateTime','ABP'], axis=1, inplace=True)
    
       
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
    
    ############median rolling filter -only to show -applied to graph
    
    #############Overall Pearson correlation -Pandas
    overall_pearson_r = dataset['BRS'].corr(dataset['ICP'])
    overall_spearman_r = dataset['BRS'].corr(dataset['ICP'], method = 'spearman')
    
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    print(f"Pandas computed Spearman r: {overall_spearman_r}")
    
    #############rolling Spearman
    r_window_size = 720
    rolling_s = rolling_spearman(dataset.BRS.to_numpy(), dataset.ICP.to_numpy(), 720)
   
    ############rolling Person
    rolling_p = dataset['BRS'].rolling(window=r_window_size, center=True, min_periods=1).corr(dataset['ICP'])
    
    #####TIME-DOMAIN CROSS CORRELATION WITH LAG
    rs = [crosscorr(dataset.BRS,dataset.ICP, lag) for lag in range(-180,180+1)]
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    
   
   
    