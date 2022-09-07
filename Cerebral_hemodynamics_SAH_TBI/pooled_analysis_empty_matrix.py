# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 23:31:58 2021

@author: agnie
"""
import os
import pandas as pd
import numpy as np

# Function to read file with spasm side
def read_file_side(file_name, separator = None):
   side = pd.read_csv (file_name, sep = separator)
   side = pd.DataFrame(side)
  
   return side




SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\wyniki_daily_RESULTS"
SOURCE_SPASM = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\side_of_vasospasm_pooled_wav.csv"
PATIENTS = 336
n_PATIENTS = 94

side = read_file_side(SOURCE_SPASM, separator = ';')
data = side["patient"]

zer = np.zeros((PATIENTS, n_PATIENTS))

for i in range (0, 94):
    wynik = data.values == i+1
    zer[:,i] = wynik
    
df_wynik = pd. DataFrame(zer)    
df_wynik.to_csv(SOURCE_FILE_PATH + '/' +'macierz_pooled.csv', sep = ';', index = False)