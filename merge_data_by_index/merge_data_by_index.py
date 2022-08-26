# -*- coding: utf-8 -*-
"""
Created on Wed May 18 08:50:08 2022

@author: agnie
"""
import pandas as pd
import numpy as np

from pathlib import Path

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\HFC_article\Summary_final_184.csv"
SOURCE_FILE_PATH_EARLY = "C:\Moje_dokumenty\Po_doktoracie_09_2021\HFC_article\params_EARLY.csv"

# Function to read file
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
                  
    return dataset


baza184 = read_file(SOURCE_FILE_PATH, separator = ';', decimal_sign = ',')

baza_early = read_file(SOURCE_FILE_PATH_EARLY, separator = ',', decimal_sign = '.')


merged_df = pd.merge(baza184, baza_early, on="FILE")

 

merged_df.to_csv(Path("C:\Moje_dokumenty\Po_doktoracie_09_2021\HFC_article\merged.csv"))