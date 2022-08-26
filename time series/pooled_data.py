# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 22:11:50 2022

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



file_name = r"C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\BRS_pooled_data.csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\wyniki"



# Function to read file
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
                  
    return dataset



dataset = read_file(file_name, separator = ';', decimal_sign = '.')
#dataset.drop(dataset.iloc[:,14:-1], axis = 1, inplace = True)
#dataset.drop(dataset.columns[-1], axis = 1, inplace = True)



empty_set = pd.DataFrame(index = np.arange(len(dataset)*len(dataset.columns)), columns =np.arange(3))
empty_set. columns = ['BRS','days','patient']

licznik = 0
pacjent = 1
days = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]

for i in range (0, len(dataset)):
    actual_raw = dataset.iloc[i,:]
    actual_raw = actual_raw.tolist()
    empty_set.iloc[licznik:licznik+len(dataset.columns),0] = actual_raw
    empty_set.iloc[licznik:licznik+len(dataset.columns),2] = pacjent
    empty_set.iloc[licznik:licznik+len(dataset.columns),1] = days
    licznik = licznik+len(dataset.columns)
    pacjent = pacjent+1
    
    
empty_set=empty_set.dropna()  

#############generacja macierzy
empty_p = pd.DataFrame(index = np.arange(len(empty_set)), columns =np.arange(len(dataset)))

for p in range (1,len(dataset)+1):
     actual_test = (empty_set['patient']==p)
     actual_test=actual_test.tolist()
     empty_p.iloc[:,p-1] =actual_test

empty_p=empty_p*1

empty_set = empty_set.reset_index(drop=True)
t = pd.concat([empty_set,empty_p], axis=1, ignore_index=True)

t.to_csv(SAVE_FILE_PATH + '/' +'pooled.csv', sep = ';', index = False)


#############

poor = [1,6,8,9,10,11,12,13,16,17,19,21,26]
good = [2,3,4,5,7,14,15,18,20,22,23,24,25,27,28,29]

empty_setP = empty_set
empty_setG = empty_set

#####################GOOOOOOD####################################
for i in poor:
    empty_setG = empty_setG.drop(empty_setG[empty_setG.patient==i].index)#########zostaje good
    
#############generacja macierzy

empty_p = pd.DataFrame(index = np.arange(len(empty_setG)), columns =np.arange(len(good)))

licznik = 0
for p in good:
     actual_test = (empty_setG['patient']==p)
     actual_test=actual_test.tolist()
     empty_p.iloc[:,licznik] =actual_test
     licznik = licznik+1

empty_p=empty_p*1

empty_setG = empty_setG.reset_index(drop=True)
tG = pd.concat([empty_setG,empty_p], axis=1, ignore_index=True)

tG.to_csv(SAVE_FILE_PATH + '/' +'pooled_good.csv', sep = ';', index = False)  
    
    
    
###############POOOOORRRRRRRRRRR

for i in good:
    empty_setP = empty_setP.drop(empty_setP[empty_setP.patient==i].index)#########zostaje poor
    
#############generacja macierzy

empty_p = pd.DataFrame(index = np.arange(len(empty_setP)), columns =np.arange(len(poor)))

licznik = 0
for p in poor:
     actual_test = (empty_setP['patient']==p)
     actual_test=actual_test.tolist()
     empty_p.iloc[:,licznik] =actual_test
     licznik = licznik+1

empty_p=empty_p*1

empty_setP = empty_setP.reset_index(drop=True)
tP = pd.concat([empty_setP,empty_p], axis=1, ignore_index=True)

tP.to_csv(SAVE_FILE_PATH + '/' +'pooled_poor.csv', sep = ';', index = False)  
   