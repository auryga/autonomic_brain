
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import statistics as st

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie\zaproszony_artykul\summary"
EXTENSION_FILE = ".csv"
PATIENTS = 1

dictionary_list =['BRS', 'ICP', 'ABP',]



               


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
        
        
 
   #############delete values less than 0 #################
    dataset[dataset< 0.001] = np.nan
   
    
    test_to_remove = dataset.copy() 
    ############# ############# ############# #############
   
         
    return dataset, test_to_remove


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
  
      

#Function to find NaN and replace by mean
def replace_NaN (data): 
    
    total_rows=len(data.axes[0])
    total_cols=len(data.axes[1])
    pusta_macierz = pd.DataFrame(index=np.arange(total_rows), columns=np.arange(total_cols))
    for i in range (0, total_cols):
        a = data.iloc[:,i]
             
        if a.isnull().sum() != len(data.index):
                      
            a = pd.DataFrame(a)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer = imputer.fit(a)
            transform_results = imputer.transform(a)
            transform_results = pd.DataFrame(transform_results)
            
            pusta_macierz.iloc[:,i]=transform_results.values
            
        
        else:
            transform_results  = a.fillna(9999)
            transform_results = pd.DataFrame(transform_results)
            pusta_macierz.iloc[:,i]=transform_results.values
        
           
            
    pusta_macierz.columns = data.columns
    return pusta_macierz


#Function to find position of variables
def find_columns(df, dictionary_list):
    
    
    counter = 0
    counter_list = 0
    dictionary_list_position = np.empty((1,len(dictionary_list)))
    dictionary_list_position[:]=np.NaN
    maska=pd.DataFrame(index=np.arange(1), columns=np.arange(len(dictionary_list)))
    maska=maska.fillna(0)
    for counter_list in range (0, len(dictionary_list)):
        for counter in range (0,len(df.columns)):
            if df.columns[counter] == dictionary_list[counter_list]:
                maska.loc[:,counter_list] =1
                dictionary_list_position[0,counter_list] =counter
                
        counter =+ 1
    counter_list += 1
    return dictionary_list_position,maska


#Function to find mean values of each signals/variables
def mean_values(data,dictionary_list,position):
   # mean_parameters = data.mean(axis = 0)
   
    results = []
    
    for i in range (0,len(dictionary_list)):
        print(i)
        if math.isnan(position[0,i])==True:
            print('nan')
            d=np.nan
        elif data.empty == True:
            d=np.nan
        else:
            print(i)
            d=data.iloc[1, int(position[0,i])].mean()
            
            
            
        results.append(d)
    
    return results

#Function to cut signals into 5-minutes windows:
def windows(data,dictionary_list,position,maska):
    ilosc_spadkow=0 #############okno nie jest overlappowane
    abp_during =[]
    icp_during=[]
    brs_during=[]
    
   
    
    abp_before =[]
    abp_before=[]
    HR_before=[]
   
    
    
    indeksy =[]
    
    rso2_left=np.nan
    abp=np.nan
    HR=np.nan
    TOX_l=np.nan
    co=np.nan
    ci=np.nan
    brs=np.nan
    
   
    
   
    
    
    if maska.iloc[0,5]==1:
        rso2_left=data.iloc[:,position[:,5]]
    if maska.iloc[0,2]==1:
        abp=data.iloc[:,position[:,2]]
    if maska.iloc[0,1]==1:
        HR=data.iloc[:,position[:,1]]
    if maska.iloc[0,6]==1:
        TOX_l=data.iloc[:,position[:,8]]
    if maska.iloc[0,4]==1:
        co=data.iloc[:,position[:,4]]
    if maska.iloc[0,3]==1:
        ci=data.iloc[:,position[:,3]]
    if maska.iloc[0,0]==1:
        brs=data.iloc[:,position[:,0]]  
        
    ######################OKNO RUCHOME###########################
    
    window=5,10, 15 -------co minute----- zalezy od próbek, co 10sekund - 6probek/minute
    #okno 5 minutowe   
    if maska.iloc[0,5]==1:
        for j in range(0,len(rso2_left),window):  ###########modyfikowalne- overalp
            mean_window=np.nanmean(ICP[j:j+window])  #srednia w oknie
            if mean_window>20:
                ilosc_spadkow+=1#############33ilosc spadków z overlapem nie ma sensu
                rso2_left_during.append(np.nanmean(rso2_left[j:j+window]))
                HR_during.append(np.nanmean(HR[j:j+window]))
                abp_during.append(np.nanmean(abp[j:j+window]))
                TOX_left_during.append(np.nanmean(TOX_l[j:j+window]))
                co_during.append(np.nanmean(co[j:j+window]))
                ci_during.append(np.nanmean(ci[j:j+window]))
                brs_during.append(np.nanmean(brs[j:j+window]))
               
                
                if j+window<len(rso2_left):
                    for i in range(j,j+window):
                        indeksy.append(i)  ##########3indeksy narostu
                    
        maska=pd.DataFrame(index=np.arange(len(rso2_left)), columns=np.arange(1))
                    
        maska=maska.replace(np.nan, 1)
        for i in indeksy:
            maska.iloc[i,:]=np.nan
                    
           
        rso2_left_during = np.nanmean(rso2_left_during)
        abp_during= np.nanmean(abp_during)
        HR_during=np.nanmean(HR_during)
        TOX_left_during=np.nanmean(TOX_left_during)
        co_during=np. nanmean(co_during)
        ci_during=np.nanmean(ci_during)
        brs_during=np.nanmean(brs_during)
        
        
        
        rso2_left_before = np.nanmean(rso2_left.iloc[:,0]*maska.iloc[:,0])
        abp_before= np.nanmean(abp.iloc[:,0]*maska.iloc[:,0])
        HR_before=np.nanmean(HR.iloc[:,0]*maska.iloc[:,0])
        TOX_left_before=np.nanmean(TOX_l.iloc[:,0]*maska.iloc[:,0])
        co_before=np.nanmean(co.iloc[:,0]*maska.iloc[:,0])
        ci_before=np.nanmean(ci.iloc[:,0]*maska.iloc[:,0])
        brs_before=np.nanmean(brs.iloc[:,0]*maska.iloc[:,0])
        

        
        
    results_during_before= pd.DataFrame(index=np.arange(1), columns=np.arange(16))

    frames = [rso2_left_during,abp_during, HR_during, TOX_left_during,
    co_during, ci_during, brs_during, ilosc_spadkow,indeksy,
    rso2_left_before,abp_before, HR_before, TOX_left_before,
    co_before, ci_before, brs_before]

        
    results_during_before.loc[len(results_during_before)] = frames
    
    results_during_before.columns=(['rso2_left_during','abp_during', 'HR_during', 'TOX_left_during',
    'co_during', 'ci_during', 'brs_during', 'ilosc_spadkow','indeksy',
    'rso2_left_before','abp_before', 'HR_before', 'TOX_left_before',
    'co_before', 'ci_before', 'brs_before'])
                
            
    return results_during_before
    



#Function to concatenate headers with values
def concatenate_list (results,dictionary_list):
    data_final = pd.DataFrame(results)
    data_final.columns = dictionary_list
    return data_final






###############################SKRYPT#####################################

        
#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)


results_iteration = []

for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset, test_to_remove = read_file(files_list[counter_files], separator = ';', decimal_sign = ',', delete_column='DateTime')

     
    
    #recall of Function to change NaN in data:
    dataset_transform = replace_NaN(dataset)
    

    #recall of Function to find headers from dictionary:
    position,maska = find_columns(dataset, dictionary_list)
    
    r=windows(dataset, dictionary_list, position,maska)
  
   
    frames = [r,ra]

    result = pd.concat(frames, axis = 1) 
    result = result.drop(labels=0, axis=0)
    results_iteration.append(result)
    
lista=['rso2_left_during','abp_during', 'HR_during', 'TOX_left_during',
    'co_during', 'ci_during', 'brs_during', 'ilosc_spadkow','indeksy',
    'rso2_left_before','abp_before', 'HR_before', 'TOX_left_before',
    'co_before', 'ci_before', 'brs_before', 'rso2_right_during','abp_during', 'HR_during', 'TOX_right_during',
    'co_during', 'ci_during', 'brs_during', 'ilosc_spadkow','indeksy',
    'rso2_right_before','abp_before', 'HR_before', 'TOX_right_before',
    'co_before', 'ci_before', 'brs_before']  

results_iteration = pd.concat(results_iteration)
size = 32

results_iteration.rename(columns={results_iteration.columns[i]: lista[i] for i in range(size)}, inplace = True)

results_iteration.to_csv(SAVE_FILE_PATH + '/' +'wzrosty20.csv', sep = ';', index = False)
 
################################

