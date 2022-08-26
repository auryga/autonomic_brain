# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:34:21 2022

@author: agnie


''''''''''''''

SKRYPT DO RYSOWANIA WYKRESÓW - PATTERN RECOGNITION
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



SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\prx_amp"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\wyniki"
korekta_file = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\korekcja.csv"




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



#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
       

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)



results_iteration = []
korekta = read_file(korekta_file, separator = ';', decimal_sign = '.')


for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ';', decimal_sign = ',')
    #columns rename
    dataset = dataset.rename(columns={"icp[mmHg]": "ICP", "abp[mmHg]": "ABP", "art": "ABP", "art[mmHg]": "ABP",
                                      "ABP_BaroIndex": "BRS", "brs": "BRS", "ART_BaroIndex": "BRS", "ART": "ABP"})
    #drop of ABP and DateTime
    dataset.drop(['DateTime','ABP'], axis=1, inplace=True)
    #drop of outliers
    dataset=outliers(dataset,3)
    
    
        
    ################FIGURE 1: graph to show raw data
    f,ax=plt.subplots(figsize=(10,5))
    ax.plot(dataset)
    ax.set(xlabel='Time',ylabel='mm Hg or ms/mm Hg')
    #ax.legend()
    
      
      
    #########interpolated
    
    filled_i = interpolate_gaps(dataset['ICP'], limit=2)
    filled_b = interpolate_gaps(dataset['BRS'], limit=2)
    
    
    dataset2 = pd.DataFrame(filled_i)
    dataset3 = pd.DataFrame(filled_b)
    frames = [dataset2,dataset3]
    result = pd.concat(frames,axis=1)
    result.columns = ['ICP', 'BRS']
    
    a = result.index/(360)
    
    ########rollling median
    result['mov_av_ICP'] = result['ICP'].rolling(window=720,center=True,min_periods=1).median()
    result[result['mov_av_ICP']<=0.1] = np.nan
    
    result['mov_av_BRS'] = result['BRS'].rolling(window=720,center=True,min_periods=1).median()
    result[result['mov_av_BRS']<=0.1] = np.nan
    
    
    
        
    # FIGURE 2: create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(10,3))
    # make a plot
    ax.plot(a,result['mov_av_ICP'], color="red", label = 'ICP')
    # set x-axis label
    ax.set_xlabel("time [hours]",fontsize=14)
    # set y-axis label
    ax.set_ylabel("ICP [mm Hg]",color = "red", fontsize=14)
    ax.set(xlim = [0,24])
    ax.set(ylim = [0,45])
     
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(a, result['mov_av_BRS'],color="blue",  label = 'BRS')
    ax2.set_ylabel("BRS [ms/mm Hg]",color="blue", fontsize=14)
    ax2.set(ylim = [0,25])
  
    plt.grid()
    plt.show()    
    
    ##########FIGURE 3
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(10,3))
    # make a plot
    ax.plot(a,result['mov_av_ICP'], color="red", label = 'ICP')
    # set x-axis label
    ax.set_xlabel("time [hours]",fontsize=14)
    #ax.set(xlim = [0, 67])
    # set y-axis label
    ax.set_ylabel("ICP [mm Hg]",color = "red", fontsize=14)
    plt.axline((0, 22), (-1, 22),color="red",linestyle = 'dashdot')
    ax.set(ylim = [0,40])
     
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(a, result['mov_av_BRS'],color="blue",  label = 'BRS')
    ax2.set_ylabel("BRS [ms/mm Hg]",color="blue", fontsize=14)
    ax2.set(ylim = [0,25])
    
  
    
    plt.grid()
    plt.show() 
    
    
    ##########FIGURE 4
    
    window = 12*360
    brs_av12 = []
    icp_av12 = []
   
    for i in range (0, len(result),window):
        brs_av12.append(np.nanmean(result['BRS'].iloc[i:i+window]))
        icp_av12.append(np.nanmean(result['ICP'].iloc[i:i+window]))
    
    a2 = [i*0.5 for i in range (1,len(brs_av12)+1)]
    
    
    ###############3korekcja na przesuwanie
    if korekta['korekcja'].iloc[counter_files] == 0:
        a2 = a2
    else:
        a2 = list(a2) + korekta['korekcja'].iloc[counter_files]
    
    
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.plot(a2,brs_av12, 'o', color='blue')
    ax.set_ylabel("BRS [ms/mm Hg]",color="blue", fontsize=14)
    ax.set_xlabel("time[days]",color="black", fontsize=14)
    #ax.set(xlim = [0,7])
   
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(a2, icp_av12,'o',color="red",  label = 'ICP')
    ax2.set_ylabel("ICP [mm Hg]",color="red", fontsize=14)
    #ax2.set(ylim = [0,40])
    fig.savefig('days_' + str(only_files_names[counter_files]) + '.png')
       
    
    ###############korelacja w dniach - TOTAL
    overall_pearson_r = result['BRS'].corr(result['ICP'])
    r, p = stats.pearsonr(dataset.dropna()['BRS'], dataset.dropna()['ICP'])
    
    
    