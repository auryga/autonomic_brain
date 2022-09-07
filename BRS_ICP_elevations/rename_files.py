# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 19:22:48 2022

@author: agnie
"""
import os
import pandas as pd
import uuid


SOURCE_FILE_PATH = r"D:\Baza_danych_OPUS_anonimised\t4"
SAVE_FILE_PATH = r"D:\Baza_danych_OPUS_anonimised\t4"
EXTENSION_FILE = ".csv"
PATIENTS = 1

          


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



#Function to get files names without extension
def get_files_names_withext(files_names):
    files_name_withext = []
    for name in range (0,len(files_names)):
        simple = os.path.splitext(files_names[name])[0]
        simple = simple[0:3]
        files_name_withext.append(simple)
        files_name_withext[name] = int(files_name_withext[name])
    return files_name_withext



# Function to read file
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
    return dataset   
        
        ###############################SKRYPT#####################################

        
#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)

N = str(uuid.uuid4())


for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset = read_file(files_list[counter_files], separator = ';', decimal_sign = ',', delete_column='DateTime')
    
    
    dataset.to_csv(SAVE_FILE_PATH + '/' + N + '_' + str(counter_files) + '.csv', sep = ';', index = False)





