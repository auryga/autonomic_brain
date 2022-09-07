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

SOURCE_SPASM = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\sides_daily_all\raveraged_withDCI_pooled.csv"
#podajemy ile wierszy zostało po usunięciu pustych
PATIENTS = 124
#liczby pacjentów nie zmieniamy
n_PATIENTS = 97

side = read_file_side(r"C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\sides_daily_all\raveraged_withDCI_pooled.csv", separator = ';')
data = side["patient"]

zer = np.zeros((PATIENTS, n_PATIENTS))

for i in range (0, 97):
    wynik = data.values == i+1
    zer[:,i] = wynik
    
df_wynik = pd. DataFrame(zer)  
df_wynik = df_wynik.loc[:, (df_wynik != 0).any(axis=0)]

  
df_wynik.to_csv(SOURCE_FILE_PATH + '/' +'macierz_pooled.csv', sep = ';', index = False)