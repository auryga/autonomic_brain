import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
#from  natsort import natsorted

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\wyniki"
PATIENTS = None

dictionary_list =['BRS', 'ICP', 'ABP']

#Function to get number of files with extension .csv:
def counter_files(path, extension):
    list_dir = []
    list_dir = os.listdir(path)
    count = 0
    for file in list_dir:
        if file.endswith(extension):
            count+=1
    PATIENTS = count
    return  count

#Function to get list of files with specific extension:
#wyciaga z lokalizacji pliki o zadanym rozszerzeniu
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
        space = simple.rfind(" - ")
        simple = simple[0:space]
        files_name_withext.append(simple)
    #files_name_withext[name] = int(files_name_withext[name])
    return files_name_withext

# Function to read file
#do file name list_files
def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset_raw = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = dataset_raw.rename(columns={"icp[mmHg]": "ICP", "abp[mmHg]": "ABP", "art": "ABP", "art[mmHg]": "ABP",
                                      "ABP_BaroIndex": "BRS", "brs": "BRS", "ART_BaroIndex": "BRS", "ART": "ABP"})
    if delete_column != None:
        del dataset[delete_column]

    dataset = pd.DataFrame(dataset)
    dataset[dataset < 0.001] = np.nan
    test_to_remove = dataset.copy()

    return dataset, test_to_remove


# Function to find position of variables
def find_columns(df, dictionary_list):
    counter = 0
    counter_list = 0
    dictionary_list_position = np.empty((1, len(dictionary_list)))
    dictionary_list_position[:] = np.NaN
    maska = pd.DataFrame(index=np.arange(1), columns=np.arange(len(dictionary_list)))
    maska = maska.fillna(0)
    for counter_list in range(0, len(dictionary_list)):
        for counter in range(0, len(df.columns)):
            if df.columns[counter] == dictionary_list[counter_list]:
                maska.loc[:, counter_list] = 1
                dictionary_list_position[0, counter_list] = counter

        counter = + 1
    counter_list += 1
    return dictionary_list_position, maska


# Function to cut signals into 5-minutes windows:
def windows(data, dictionary_list, position, maska):

    abp_before = []
    brs_before = []
    icp_before = []

    abp_during = []
    brs_during = []
    icp_during = []

    indeksy = []

    abp = np.nan
    brs = np.nan
    icp = np.nan


    if maska.iloc[0, 2] == 1:
        abp = data.iloc[:, position[:, 2]]
    if maska.iloc[0, 0] == 1:
        brs = data.iloc[:, position[:, 0]]
    if maska.iloc[0, 1] == 1:
        icp = data.iloc[:, position[:, 1]]

        ######################OKNO RUCHOME###########################

    window = 6*360*1
    overlap = 6*360*1
    "-------co minute----- zalezy od próbek, co 10sekund - 6probek/minute czyli 30"
    # okno 5 minutowe0
    if maska.iloc[0, 1] == 1:
        for j in range(0, len(icp), overlap):  ###########modyfikowalne- overalp
            mean_window = np.nanmean(icp[j:j + window])  # srednia w oknie
            if mean_window > 20:
                abp_during.append(np.nanmean(abp[j:j + window]))
                brs_during.append(np.nanmean(brs[j:j + window]))
                icp_during.append(np.nanmean(icp[j:j + window]))

                if j + window < len(icp):
                    for i in range(j, j + window):
                        indeksy.append(i)  ##########3indeksy narostu

            mean_window_before = mean_window


        maska = pd.DataFrame(index=np.arange(len(icp)), columns=np.arange(1), dtype=int)

        #zmiana nan na 1, w wierszu z indeksem i (tam sa indeksy podczas trwania narostu) wszystkie kolumny na nan
        maska = maska.replace(np.nan, 1)
        for i in indeksy:
            maska.iloc[i, :] = np.nan

        abp_during = np.nanmean(abp_during)
        brs_during = np.nanmean(brs_during)
        icp_during = np.nanmean(icp_during)

        abp_before = np.nanmean(abp.iloc[:, 0] * maska.iloc[:, 0])
        brs_before = np.nanmean(brs.iloc[:, 0] * maska.iloc[:, 0])
        icp_before = np.nanmean(icp.iloc[:, 0] * maska.iloc[:, 0])

    results_during_before = pd.DataFrame(index=np.arange(1), columns=np.arange(6))

    frames = [abp_during, brs_during, icp_during, abp_before, brs_before, icp_before]

    results_during_before.loc[len(results_during_before)] = frames

    results_during_before.columns = (['abp_during', 'brs_during', 'icp_during', 'abp_before',
                                      'brs_before', 'icp_before'])

    return results_during_before


# Function to concatenate headers with values
def concatenate_list(results, dictionary_list):
    data_final = pd.DataFrame(results)
    data_final.columns = dictionary_list
    return data_final

#function to find extreme values
def outliers(data, ex):
    total_cols=len(data.axes[1])
    
    for i in range (0, total_cols):
        kolumna = data.iloc[:,i]
        for j in range (0, len(kolumna)):
            if (kolumna[j] == 9999):
                kolumna[j] = None 
           
        q1 = kolumna.quantile(q =0.25)
        q3 = kolumna.quantile(q = 0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-ex*iqr
        fence_high = q3+ex*iqr
        df_out = kolumna.loc[(kolumna < fence_low )|(kolumna > fence_high)]
        kolumna[df_out.index] = None
        
    return data





#####################KOD WLASCIWY###################

files_number = counter_files(SOURCE_FILE_PATH, EXTENSION_FILE)
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH, EXTENSION_FILE)

patient_list = get_files_names_withext(only_files_names)


results_iteration = []



for i in range(0,files_number):

   
    data, test_to_remove = read_file(files_list[i], separator = ',', decimal_sign = '.', delete_column='DateTime')
    data = outliers(data,3)
    
    
    position, maska = find_columns(data, dictionary_list)

    r = windows(data, dictionary_list, position, maska)

    frames = [r]

    result = pd.concat(frames, axis=1)
    result = result.drop(labels=0, axis=0)
    results_iteration.append(result)

lista = ['abp_during', 'brs_during', 'icp_during',
         'abp_before', 'brs_before', 'icp_before']
size = len(lista)

results_iteration = pd.concat(results_iteration)
print(type(results_iteration))


results_iteration.rename(columns={results_iteration.columns[i]: lista[i] for i in range(size)}, inplace=True)
results_iteration['patient'] = patient_list
print(results_iteration)
results_iteration.to_csv(SAVE_FILE_PATH + '/' + 'cisnienie22.csv', sep=',', index=False)




