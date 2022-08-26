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


SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\wyniki"
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
brs_av12_values_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))
icp_av12_values_iteration = pd.DataFrame(index = np.arange(files_number), columns =np.arange(26))

korekcja = read_file(wsp_korekcji, separator = ';', decimal_sign = '.')

for counter_files in range(0,files_number):

    
    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ',', decimal_sign = '.')
    #columns rename
    dataset = dataset.rename(columns={"icp[mmHg]": "ICP", "abp[mmHg]": "ABP", "art": "ABP", "art[mmHg]": "ABP",
                                      "ABP_BaroIndex": "BRS", "brs": "BRS", "ART_BaroIndex": "BRS", "ART": "ABP"})
    #drop of ABP and DateTime
    dataset.drop(['DateTime','ABP'], axis=1, inplace=True)
    #drop of outliers
    dataset=outliers(dataset,3)
    
    
       
      
    #########interpolated
    
    filled_i = interpolate_gaps(dataset['ICP'], limit=2)
    filled_b = interpolate_gaps(dataset['BRS'], limit=2)
    
    
    dataset2 = pd.DataFrame(filled_i)
    dataset3 = pd.DataFrame(filled_b)
    frames = [dataset2,dataset3]
    result = pd.concat(frames,axis=1)
    result.columns = ['ICP', 'BRS']
    
    a = result.index/360
    a_k = result.index/(360*24)
    
    if korekcja['korekcja'].loc[counter_files]!=0:
        a_k = a_k+korekcja['korekcja'].loc[counter_files]
    
    
    
    ########rollling median
    result['mov_av_ICP'] = result['ICP'].rolling(window=720,center=True,min_periods=1).median()
    result[result['mov_av_ICP']<=0.1] = np.nan
    
    result['mov_av_BRS'] = result['BRS'].rolling(window=720,center=True,min_periods=1).median()
    result[result['mov_av_BRS']<=0.1] = np.nan
    
     
    
    
    ###############korelacja w dniach - TOTAL
    r = result['BRS'].corr(result['ICP'],method = 'spearman')
    rx, p = stats.pearsonr(dataset.dropna()['BRS'], dataset.dropna()['ICP'])
    
    results_iteration['r_total'].loc[counter_files] = r
    results_iteration['p_total'].loc[counter_files] = p
    
    ##############korelacja w pierwszej dobie
    
    first_day = 360*24
    if len(result)<=first_day:
         brs_first_day = result['BRS']
         icp_first_day = result['ICP']
    else:
        brs_first_day = result['BRS'].iloc[0:first_day+1]
        icp_first_day = result['ICP'].iloc[0:first_day+1]
        
    brs_first_day = pd.DataFrame(brs_first_day) 
    icp_first_day = pd.DataFrame(icp_first_day) 
    
    #rs1 =   brs_first_day.corr(icp_first_day)
    rs1 = brs_first_day['BRS'].corr(icp_first_day['ICP'], method = 'spearman')
    
    results_iteration['r_1'].loc[counter_files] = rs1
    
    #####TIME-DOMAIN CROSS CORRELATION WITH LAG - TOTAL without 1 doba
    BRS_total_bez1 = result['BRS'].iloc[[360*24:-1],:]
    ICP_total_bez1 = result['ICP'].iloc[360*24:-1,:]
    rs = [crosscorr(BRS_total_bez1,ICP_total_bez1, lag) for lag in range(0,180+1)]
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    offset_minutes = offset/6
    
    results_iteration['offset_total'].loc[counter_files] = offset_minutes
    
    #####TIME-DOMAIN CROSS CORRELATION WITH LAG - FIRST DAY
    rs_1d = [crosscorr(brs_first_day['BRS'],icp_first_day['ICP'], lag) for lag in range(0,180+1)]
    offset_1d = np.floor(len(rs_1d)/2)-np.argmax(rs_1d)
    offset_1d_minutes = offset_1d/6
    results_iteration['offset_1'].loc[counter_files] = offset_1d_minutes
    
    ##################FIGURES
    # FIGURE 1: create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(20,10))
    # make a plot
    ax.plot(a,result['mov_av_ICP'], color="red", label = 'ICP')
    # set x-axis label
    ax.set_xlabel("time [hours]")
    # set y-axis label
    ax.set_ylabel("ICP [mm Hg]",color = "red")
    plt.axline((0, 22), (-1, 22),color="red",linestyle = 'dashdot')
    ax.set(xlim = [0,24])
    ax.set(ylim = [0,40])
    ax.set(title=f'rs = {rs1:.2f}')
    plt.grid()
    
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(a, result['mov_av_BRS'],color="blue",  label = 'BRS')
    ax2.set_ylabel("BRS [ms/mm Hg]",color="blue")
    ax2.set(ylim = [0,45])
  
    
       
    plt.savefig('first_day'+only_files_names[counter_files]+".jpg")
    
     
    
    
    
    
    ##########FIGURE 2
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots(figsize=(20,10))
    # make a plot
    ax.plot(a_k,result['mov_av_ICP'], color="red", label = 'ICP')
    # set x-axis label
    ax.set_xlabel("time [days]")
    #ax.set(xlim = [0, 67])
    # set y-axis label
    ax.set_ylabel("ICP [mm Hg]",color = "red")
    plt.axline((0, 22), (-1, 22),color="red",linestyle = 'dashdot')
    ax.set(ylim = [0,40])
    plt.xlim(left=0)
    ax.set(title=f'rs = {r:.2f}')
    plt.grid()
    
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(a_k, result['mov_av_BRS'],color="blue",  label = 'BRS')
    ax2.set_ylabel("BRS [ms/mm Hg]",color="blue")
    ax2.set(ylim = [0,45])
    plt.xlim(left=0)
    
  
    
    
    plt.savefig('total_days'+only_files_names[counter_files]+".jpg")
    
    
    ##########FIGURE 4
    
    window = 12*360
    brs_av12 = []
    icp_av12 = []
   
    for i in range (0, len(result),window):
        brs_av12.append(np.nanmean(result['BRS'].iloc[i:i+window]))
        icp_av12.append(np.nanmean(result['ICP'].iloc[i:i+window]))
    
    a2 = [i*0.5 for i in range (1,len(brs_av12)+1)]
    
    if korekcja['korekcja'].loc[counter_files]!=0:
        a2 = a2+korekcja['korekcja'].loc[counter_files]
    
    
    
    
    fig,ax = plt.subplots(figsize=(10,10))
    ax.plot(a2,brs_av12, 'o', color='blue')
    ax.set_ylabel("BRS [ms/mm Hg]",color="blue")
    ax.set_xlabel("time[days]",color="black")
    ax.set(ylim = [0,30])
    
    
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(a2, icp_av12,'o',color="red",  label = 'ICP')
    ax2.set_ylabel("ICP [mm Hg]",color="red")
    ax2.set(ylim = [0,40])
    
    plt. savefig('following_days'+only_files_names[counter_files]+".jpg")
    
    #norm_brs_12 = []
    #for i in range (0, len(brs_av12)):
        #norm_brs_12.append ((brs_av12[i]-min(brs_av12))/(max(brs_av12)-min(brs_av12)))
    
    #norm_icp_12 = []
    #for i in range (0, len(icp_av12)):
        #norm_icp_12.append ((icp_av12[i]-min(icp_av12))/(max(icp_av12)-min(icp_av12)))
    
    
    #fig,ax = plt.subplots(figsize=(10,10))
    #ax.plot(a2,norm_brs_12, linestyle='solid', color='blue')
    #ax.set_ylabel("normalised BRS",color="blue")
    #ax.set_xlabel("time[days]",color="black")
    
    
    
    # twin object for two different y-axis on the sample plot
    #ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    #ax2.plot(a2, norm_icp_12,linestyle='solid',color="red",  label = 'ICP')
    #ax2.set_ylabel("normalised ICP",color="red")
    
    #plt. savefig('following_days_normalised'+only_files_names[counter_files]+".jpg")
    
    
    #brs_av12_iteration.iloc[counter_files,0:len(norm_brs_12)] = norm_brs_12
    #icp_av12_iteration.iloc[counter_files,0:len(norm_icp_12)] = norm_icp_12
    brs_av12_values_iteration.iloc[counter_files,0:len(brs_av12)] = brs_av12
    icp_av12_values_iteration.iloc[counter_files,0:len(icp_av12)] = icp_av12
    
#############przesuwanie macierzy
brs_av12_korekcja = pd.DataFrame(index = np.arange(files_number), columns =np.arange(40))
brs_av12_korekcja.iloc[:,len(brs_av12_values_iteration)] = brs_av12_values_iteration

icp_av12_korekcja = pd.DataFrame(index = np.arange(files_number), columns =np.arange(40))
icp_av12_korekcja.iloc[:,len(icp_av12_values_iteration)] = icp_av12_values_iteration

for counter_files in range(0,files_number):

    korekcja_aktualna = korekcja['korekcja'].loc[counter_files]

    if korekcja['korekcja'].loc[counter_files]!=0:
        z = brs_av12_values_iteration.iloc[counter_files,0:len(brs_av12_values_iteration.columns)]
        z = z.tolist()
        brs_av12_korekcja.iloc[counter_files,(korekcja_aktualna*2):(len(brs_av12_values_iteration.columns)+korekcja_aktualna*2)] = z
        
        zi = icp_av12_values_iteration.iloc[counter_files,0:len(icp_av12_values_iteration.columns)]
        zi = zi.tolist()
        icp_av12_korekcja.iloc[counter_files,(korekcja_aktualna*2):(len(icp_av12_values_iteration.columns)+korekcja_aktualna*2)] = zi
        
             
                
    else:
        z = brs_av12_values_iteration.iloc[counter_files,0:len(brs_av12_values_iteration.columns)]
        z = z.tolist()
        brs_av12_korekcja.iloc[counter_files,0:len(z)] =  z
        
        
        zi = icp_av12_values_iteration.iloc[counter_files,0:len(icp_av12_values_iteration.columns)]
        zi = zi.tolist()
        icp_av12_korekcja.iloc[counter_files,0:len(zi)] =  zi


    
results_iteration.to_csv(SAVE_FILE_PATH + '/' +'all_files.csv', sep = ';', index = False)
#brs_av12_iteration.to_csv(SAVE_FILE_PATH + '/' +'brs_av12.csv', sep = ';', index = False)
#icp_av12_iteration.to_csv(SAVE_FILE_PATH + '/' +'icp_av12.csv', sep = ';', index = False)
brs_av12_values_iteration.to_csv(SAVE_FILE_PATH + '/' +'brs_av12_values.csv', sep = ';', index = False)
brs_av12_korekcja.to_csv(SAVE_FILE_PATH + '/' +'brs_av12_korekcja.csv', sep = ';', index = False)
icp_av12_korekcja.to_csv(SAVE_FILE_PATH + '/' +'icp_av12_korekcja.csv', sep = ';', index = False)

#############3transpozycja wyników do pooled_data
licznik = 0
days_BRS = []
days_BRS_values = []

#for i in range (0, len(brs_av12_values_iteration.columns)):
for i in range (0, len(brs_av12_korekcja.columns)):    
    #days_BRS_values.append(brs_av12_values_iteration.iloc[:,i])
    days_BRS_values.append(brs_av12_korekcja.iloc[:,i])
flatList_values = [ item for elem in days_BRS_values for item in elem]

licznik = 0
flatdays = pd.DataFrame(index = np.arange(len(flatList_values)),columns = np.arange(1))
for i in range (0,len(flatList_values),files_number):
    flatdays.iloc[i:i+files_number]=licznik+0.5
    licznik = licznik+0.5

flat_all = pd.concat([flatdays, pd.DataFrame(flatList_values), pd.DataFrame(flatList_values)], axis=1)

flat_all.to_csv(SAVE_FILE_PATH + '/' +'flat_all.csv', sep = ';', index = False)
