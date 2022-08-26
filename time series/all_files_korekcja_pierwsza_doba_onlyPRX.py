# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:33:54 2022

@author: agnie
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 14:34:21 2022

OBLICZANIE :
    R SPEARMANA
    TLCC

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

import re



SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\prx_amp"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\prx_amp"
wsp_korekcji = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\korekcja.csv"




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

plt.rcParams['font.size'] = 16

#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)

results_iteration = pd.DataFrame(index = np.arange(files_number), columns = np.arange(5))
results_iteration.columns =['r_1','r_total','p_total','offset_1', 'offset_total']

#brs_av12_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))
#icp_av12_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))

prx_av12_values_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))
rap_av12_values_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))
amp_av12_values_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))
icp_av12_values_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))

korekcja = read_file(wsp_korekcji, separator = ';', decimal_sign = '.')

for counter_files in range(0,files_number):

    
    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ';', decimal_sign = ',')
    #columns rename
    dataset = dataset.rename(columns={"icp[mmHg]": "ICP", "abp[mmHg]": "ABP", "art": "ABP", "art[mmHg]": "ABP",
                                      "ABP_BaroIndex": "BRS", "brs": "BRS", "ART_BaroIndex": "BRS", "ART": "ABP"})
    #drop of ABP and DateTime
    dataset.drop(['DateTime'], axis=1, inplace=True)
    
    

    #drop of outliers
    dataset=outliers(dataset,3)
    
    
       
      
    #########interpolated
    
    filled_p = interpolate_gaps(dataset['PRx'], limit=2)
    filled_r = interpolate_gaps(dataset['RAP'], limit=2)
    filled_am = interpolate_gaps(dataset['ICP_FundAmp'], limit=2)
    filled_icp = interpolate_gaps(dataset['ICP'], limit=2)
    
          
    dataset2 = pd.DataFrame(filled_p)
    dataset3 = pd.DataFrame(filled_r)
    dataset4 = pd.DataFrame(filled_am)
    dataset5 = pd.DataFrame(filled_icp)
   
    frames = [dataset2, dataset3, dataset4, dataset5]
    result = pd.concat(frames,axis=1)
    result.columns = ['PRx', 'RAP', 'AMP', 'ICP']
    
    
    
    
    
    ##########FIGURE 4
    
    window = 12*360
   
    prx_av12 = []
    rap_av12 = []
    amp_av12 = []
    icp_av12 = []
   
    for i in range (0, len(result),window):
       
        prx_av12.append(np.nanmean(result['PRx'].iloc[i:i+window]))
        rap_av12.append(np.nanmean(result['RAP'].iloc[i:i+window]))
        amp_av12.append(np.nanmean(result['AMP'].iloc[i:i+window]))
        icp_av12.append(np.nanmean(result['ICP'].iloc[i:i+window]))
    
    a2 = [i*0.5 for i in range (1,len(prx_av12)+1)]
    
    if korekcja['korekcja'].loc[counter_files]!=0:
        a2 = a2+korekcja['korekcja'].loc[counter_files]
    
    
        ###############prx
    fig,ax = plt.subplots(figsize=(10,10))
    ax.plot(a2,prx_av12, 'o', color='blue')
    ax.set_ylabel("PRx",color="blue")
    ax.set_xlabel("time[days]",color="black")
    #ax.set(ylim = [0,30])
       
    plt.savefig('PRx_'+only_files_names[counter_files]+".jpg")
    
       
    
    
    
    
    prx_av12_values_iteration.iloc[counter_files,0:len(prx_av12)] = prx_av12
    rap_av12_values_iteration.iloc[counter_files,0:len(rap_av12)] = rap_av12
    amp_av12_values_iteration.iloc[counter_files,0:len(amp_av12)] = amp_av12
    icp_av12_values_iteration.iloc[counter_files,0:len(icp_av12)] = icp_av12
    
#############przesuwanie macierzy
prx_av12_korekcja = pd.DataFrame(index = np.arange(files_number), columns =np.arange(40))
prx_av12_korekcja.iloc[:,len(prx_av12_values_iteration)] = prx_av12_values_iteration

rap_av12_korekcja = pd.DataFrame(index = np.arange(files_number), columns =np.arange(40))
rap_av12_korekcja.iloc[:,len(rap_av12_values_iteration)] = rap_av12_values_iteration

amp_av12_korekcja = pd.DataFrame(index = np.arange(files_number), columns =np.arange(40))
amp_av12_korekcja.iloc[:,len(amp_av12_values_iteration)] = amp_av12_values_iteration

icp_av12_korekcja = pd.DataFrame(index = np.arange(files_number), columns =np.arange(40))
icp_av12_korekcja.iloc[:,len(icp_av12_values_iteration)] = icp_av12_values_iteration



for counter_files in range(0,files_number):

    korekcja_aktualna = korekcja['korekcja'].loc[counter_files]

    if korekcja['korekcja'].loc[counter_files]!=0:
        z = prx_av12_values_iteration.iloc[counter_files,0:len(prx_av12_values_iteration.columns)]
        z = z.tolist()
        prx_av12_korekcja.iloc[counter_files,(korekcja_aktualna*2):(len(prx_av12_values_iteration.columns)+korekcja_aktualna*2)] = z
        
        zi = rap_av12_values_iteration.iloc[counter_files,0:len(rap_av12_values_iteration.columns)]
        zi = zi.tolist()
        rap_av12_korekcja.iloc[counter_files,(korekcja_aktualna*2):(len(rap_av12_values_iteration.columns)+korekcja_aktualna*2)] = zi
        
        zj = amp_av12_values_iteration.iloc[counter_files,0:len(amp_av12_values_iteration.columns)]
        zj = zj.tolist()
        amp_av12_korekcja.iloc[counter_files,(korekcja_aktualna*2):(len(amp_av12_values_iteration.columns)+korekcja_aktualna*2)] = zj
        
        zk = icp_av12_values_iteration.iloc[counter_files,0:len(icp_av12_values_iteration.columns)]
        zk = zk.tolist()
        icp_av12_korekcja.iloc[counter_files,(korekcja_aktualna*2):(len(icp_av12_values_iteration.columns)+korekcja_aktualna*2)] = zk
             
                
    else:
        z = prx_av12_values_iteration.iloc[counter_files,0:len(prx_av12_values_iteration.columns)]
        z = z.tolist()
        prx_av12_korekcja.iloc[counter_files,0:len(z)] =  z
        
        
        zi = rap_av12_values_iteration.iloc[counter_files,0:len(rap_av12_values_iteration.columns)]
        zi = zi.tolist()
        rap_av12_korekcja.iloc[counter_files,0:len(zi)] =  zi
        
        zj = amp_av12_values_iteration.iloc[counter_files,0:len(amp_av12_values_iteration.columns)]
        zj = zj.tolist()
        amp_av12_korekcja.iloc[counter_files,0:len(zj)] =  zj
        
        zk = icp_av12_values_iteration.iloc[counter_files,0:len(icp_av12_values_iteration.columns)]
        zk = zk.tolist()
        icp_av12_korekcja.iloc[counter_files,0:len(zj)] =  zk
        
        
        
       

    
#results_iteration.to_csv(SAVE_FILE_PATH + '/' +'all_files.csv', sep = ';', index = False)
#brs_av12_iteration.to_csv(SAVE_FILE_PATH + '/' +'brs_av12.csv', sep = ';', index = False)
#icp_av12_iteration.to_csv(SAVE_FILE_PATH + '/' +'icp_av12.csv', sep = ';', index = False)
#brs_av12_values_iteration.to_csv(SAVE_FILE_PATH + '/' +'brs_av12_values.csv', sep = ';', index = False)
prx_av12_korekcja.to_csv(SAVE_FILE_PATH + '/' +'prx_av12_korekcja.csv', sep = ';', index = False)
rap_av12_korekcja.to_csv(SAVE_FILE_PATH + '/' +'rap_av12_korekcja.csv', sep = ';', index = False)
amp_av12_korekcja.to_csv(SAVE_FILE_PATH + '/' +'amp_av12_korekcja.csv', sep = ';', index = False)
icp_av12_korekcja.to_csv(SAVE_FILE_PATH + '/' +'icp_av12_korekcja.csv', sep = ';', index = False)