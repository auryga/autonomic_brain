import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import csv
#from  natsort import natsorted

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\wyniki"
PATIENTS = None
OUTCOME = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\outcome2.csv"

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
    #dataset[dataset < 0.001] = np.nan
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
def windows(data, dictionary_list, position, maska, metadata,i):
    outcome = metadata['outcome'].tolist()
    mortality = metadata['mortality'].tolist()
    age = metadata['age'].tolist()
    GCS = metadata['GCS'].tolist()
    ID = metadata['ID'].tolist()


    if maska.iloc[0, 2] == 1:
        abp = data.iloc[:, position[:, 2]]
    if maska.iloc[0, 0] == 1:
        brs = data.iloc[:, position[:, 0]]
    if maska.iloc[0, 1] == 1:
        icp = data.iloc[:, position[:, 1]]

        ######################OKNO RUCHOME###########################

    window = 90
    overlap = 45
    narosty_icp = []
    spadek_icp = []
    narosty_brs = []
    spadek_brs = []
    outcome_s = []
    ID_s = []
    mortality_s = []
    age_s = []
    GCS_s = []
    "-------co minute----- zalezy od próbek, co 10sekund - 6probek/minute czyli 30"
    # okno 5 minutowe0
    if maska.iloc[0, 1] == 1:
        treshold = 20
        
        for j in range(0, len(icp), overlap):  ###########modyfikowalne- overalp
            if j>0:
            
                mean_window = np.nanmean(icp[j:j + window])  # srednia w oknie
            
                if mean_window >= treshold:
                    if (np.nanmean(icp[j-window:j])) < treshold-5:
                        narosty_icp.append(np.nanmean(icp[j:j + window]))
                        spadek_icp.append(np.nanmean(icp[j-window:j]))
                        narosty_brs.append(np.nanmean(brs[j:j + window]))
                        spadek_brs.append(np.nanmean(brs[j-window:j]))
                        outcome_s.append(outcome[i])
                        ID_s.append(ID[i])
                        mortality_s.append(mortality[i])
                        age_s.append(age[i])
                        GCS_s.append(GCS[i])
                else:
                    if mean_window < treshold - 5:
                        if (np.nanmean(icp[j-1:j-1 + window])) >= treshold:
                                              
                            narosty_icp.append(np.nanmean(icp[j-window:j]))
                            spadek_icp.append(np.nanmean(icp[j:j + window]))
                            narosty_brs.append(np.nanmean(brs[j-window:j]))
                            spadek_brs.append(np.nanmean(brs[j:j + window]))
                            outcome_s.append(outcome[i])
                            ID_s.append(ID[i])
                            mortality_s.append(mortality[i])
                            age_s.append(age[i])
                            GCS_s.append(GCS[i])
            
           

    results_during_before = pd.DataFrame(index=np.arange(1), columns=np.arange(9))

    frames = [narosty_icp, spadek_icp, narosty_brs, spadek_brs, outcome, ID, mortality, age, GCS]

    results_during_before.loc[len(results_during_before)] = frames

    results_during_before.columns = (['narosty_icp', 'spadek_icp', 'narosty_brs', 'spadek_brs', 'outcome','ID', 'mortality', 'age', 'GCS'])

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

metadata = pd.read_csv(OUTCOME, sep = ";")
metadata['ID'] = patient_list


#outcome = metadata['outcome'].tolist()
#mortality = metadata['mortality'].tolist()
#age = metadata['age'].tolist()
#GCS = metadata['GCS'].tolist()

results_iteration = []



for i in range(0,files_number):

   
    data, test_to_remove = read_file(files_list[i], separator = ',', decimal_sign = '.', delete_column='DateTime')
    data = outliers(data,3)
    
    
    position, maska = find_columns(data, dictionary_list)

    r = windows(data, dictionary_list, position, maska,metadata,i)

    frames = [r]

    result = pd.concat(frames, axis=1)
    result = result.drop(labels=0, axis=0)
    results_iteration.append(result)
    

#lista = ['narosty_icp', 'spadek_icp', 'narosty_brs', 'spadek_brs']
#size = len(lista)

results_iteration = pd.concat(results_iteration)




narost_brs = results_iteration["narosty_brs"].apply(pd.Series)
spadki_brs = results_iteration["spadek_brs"].apply(pd.Series)
narost_icp = results_iteration["narosty_icp"].apply(pd.Series)
spadki_icp = results_iteration["spadek_icp"].apply(pd.Series)
ID  = results_iteration['ID'].apply(pd.Series)
outcome = results_iteration ['outcome'].apply(pd.Series)
mortality = results_iteration ['mortality'].apply(pd.Series)
age = results_iteration ['age'].apply(pd.Series)
GCS = results_iteration ['GCS'].apply(pd.Series)


def pooled(data_frame):
    size = len(data_frame.columns)
    matrix = data_frame.to_numpy()
    matrix_pooled = np.concatenate(np.hsplit(matrix,size))
    matrix_pooled = pd.DataFrame(matrix_pooled)
    return matrix_pooled
        
narosty_BRS = pooled(narost_brs) 
spadki_BRS = pooled(spadki_brs)
narosty_ICP = pooled(narost_icp)
spadki_ICP = pooled(spadki_icp)
ID = pooled(ID)
outcome = pooled(outcome)
mortality = pooled(mortality)
age = pooled(age)
GCS =pooled(GCS)
 
       
    


result = pd.concat([narosty_BRS, spadki_BRS, narosty_ICP, spadki_ICP, ID, outcome, mortality, age, GCS], axis=1)
result.columns = ['narosty_BRS', 'spadki_BRS', 'narosty_ICP', 'spadki_ICP', 'ID', 'outcome', 'mortality', 'age', 'GCS']
r = result.dropna()




#results_iteration.rename(columns={results_iteration.columns[i]: lista[i] for i in range(size)}, inplace=True)
#results_iteration['patient'] = patient_list

r.to_csv(SAVE_FILE_PATH + '/' + 'okna.csv', sep=',', index=False)




