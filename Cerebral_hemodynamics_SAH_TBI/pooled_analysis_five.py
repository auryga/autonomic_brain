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
SOURCE_SPASM = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\side_of_vasospasm_pooled.csv"
SOURCE_POOLED = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\results_pooled.csv"
PATIENTS = 339
PATIENTS_n = 97


side = read_file_side(SOURCE_SPASM, separator = ';')
dane_obliczone = read_file_side(r"C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\results_pooled.csv", separator = ';')
#ilosc kolumn
co = len(dane_obliczone.columns)
patient = side["patient"]
days_patients = side ["Day_post_SAH"]

#pusta macierz do stworzenia maski binarnej
zer = np.zeros((PATIENTS, 1))
lista_tr = np.zeros((PATIENTS, 1))

df_wynik = pd.DataFrame(index=np.arange(PATIENTS_n), columns=np.arange(co)) 
df_wynik.columns = dane_obliczone.columns

for i in range (0, 97):
   wynik = patient.values == i+1
   bufor_dni = days_patients[wynik]
   b = bufor_dni<30
   b=pd.DataFrame(b)
   for j in range (0, co):
       df_wynik.iloc[i,j] = np.nanmean(dane_obliczone.iloc[b.index[b['Day_post_SAH']],j])
 
   
#zer = pd.DataFrame(zer)
#zer.columns=['Kolumna'] 
#X = zer.index[zer['Kolumna'] == 1].tolist() 


#Z= dane_obliczone.loc[X,:] 

  
df_wynik = pd. DataFrame(df_wynik)    
df_wynik.to_csv(SOURCE_FILE_PATH + '/' +'seven_recordings.csv', sep = ';')