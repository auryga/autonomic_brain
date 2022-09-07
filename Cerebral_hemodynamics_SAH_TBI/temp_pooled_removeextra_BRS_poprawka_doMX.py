
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie\TAU_DCI_SKURCZ\BRS"
SOURCE_SPASM = "C:\Moje_dokumenty\Po_doktoracie\TAU_DCI_SKURCZ\spasm\side_of_vasospasm_pooled.csv"
EXTENSION_FILE = ".csv"
PATIENTS = 97

dictionary_list =['HRV_RMSSD', 'BRS', 'HR', 'HRhfRel', 'HRV_HF','HRV_LF', 'HRV_LFHF', 'HRV_TP', 'HRV_SDSD','HRlfDivhtNorm',  'HRlfDivhtRel', 'HRhfN', 'ABP_MEAN', 'Mx_L', 'Mx_R', 'FVR', 'FVL']
dictionary_list_test =['HRV_RMSSD', 'BRS', 'HR', 'HRhfRel', 'HRV_HF','HRV_LF', 'HRV_LFHF', 'HRV_TP', 'HRV_SDSD','HRlfDivhtNorm',  'HRlfDivhtRel', 'HRhfN', 'BP_MEAN', 'Mx_L', 'Mx_R', 'FVR', 'FVL','BRS_Mx'] 

   
ipsi_list = ['FV_ipsi', 'FV_contra', 'Mx_ipsi', 'Mx_contra']
    
               


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
def read_file(file_name, separator = None , decimal_sign =None , delete_column = None ):
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
        df_out = kolumna.loc[(kolumna < fence_low )| (kolumna > fence_high)]
        a[df_out.index] = None
        
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
    dictionary_list_position = np.empty((1,17))
    dictionary_list_position[:]=np.NaN
    
    for counter_list in range (0, len(dictionary_list)):
        for counter in range (0,len(df.columns)):
            if df.columns[counter] == dictionary_list[counter_list]:
                dictionary_list_position[0,counter_list] =counter
        counter =+ 1
    counter_list += 1
    return dictionary_list_position


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

#Function to concatenate headers with values
def concatenate_list (results,dictionary_list):
    data_final = pd.DataFrame(results)
    data_final.columns = dictionary_list
    return data_final



# Function to read file with spasm side
def read_file_side(file_name, separator = None):
   side = pd.read_csv (file_name, sep = separator)
   side = pd.DataFrame(side)
  
   return side

# Function to add new empty columns with 'ipsi'/'contra'
def adding_new_ipsi_columns(ipsi_list, results_final):
    ipsi_number = len(ipsi_list)
    for ipsi_columns in range (0, ipsi_number):
        results_final[ipsi_list[ipsi_columns]] = np.nan


# # Function to remove left and right columns
def remove_LR_columns (remove_list,data):
    del_number = len(remove_list)
    for n in range (0, del_number):
       del data[remove_list[n]] 





        
#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)

side = read_file_side(SOURCE_SPASM, separator = ';')

#recall of Function to get files names without extension:
#list_without_ext = get_files_names_withext(only_files_names)

results_iteration = []
for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset, test_to_remove = read_file(files_list[counter_files], separator = ';', decimal_sign = ',', delete_column = 'DateTime')

    #recall of Function to change NaN in data:
    dataset_transform = replace_NaN(dataset)
    
    
    #dataset_transform.abp[(dataset_transform['abp'] < 50) | (dataset_transform['abp'] > 250) ] = np.nan
    #dataset_transform.FVL[(dataset_transform['FVL'] < 20) | (dataset_transform['FVL'] > 250) ]=np.nan
    #dataset_transform.FVR[(dataset_transform['FVR'] < 20) | (dataset_transform['FVL'] > 250) ]=np.nan
   
    
    
    

    #recall of Function to find headers from dictionary:
    position = find_columns(dataset_transform, dictionary_list)
    
    
    
    #recall of Function to calculate mean values
    results = mean_values(dataset_transform, dictionary_list, position)
    
    
    #BRS_AUTO to get values only where Mx is monitored
    
    M_registred=pd.DataFrame(index=np.arange(len(dataset.index)), columns=np.arange(1))
    M_registred=M_registred.fillna(0)
    
    if (math.isnan(position[0,13]) == False):
        Mx_columnL = position[0,13]
        MxL=dataset.iloc[:,int(Mx_columnL)]
        MxL=MxL.to_frame()
        for i in range (0, len(dataset.index)):
            if math.isnan(MxL.iloc[i,0])==False:
                M_registred.iloc[i,0]=1
        
        
    if (math.isnan(position[0,14]) == False):
        Mx_columnR = position[0,14]
        MxR=dataset.iloc[:,int(Mx_columnR)]
        MxR=MxR.to_frame()
        for i in range (0, len(dataset.index)):
            if math.isnan(MxR.iloc[i,0])==False:
                M_registred.iloc[i,0]=1
    
    BRS_Mx=pd.DataFrame(index=np.arange(len(dataset.index)), columns=np.arange(1))
     
    if (math.isnan(position[0,1]) == False):   
        for i in range (0, len(dataset.index)):
            BRS_Mx.iloc[i,0] = (dataset.iloc[i,int(position[0,1])]) * M_registred.iloc[i,0]                  
        BRS_Mx=BRS_Mx.replace(0,np.nan)   

    if BRS_Mx.isnull().values.any()==False:              
       BRS_Mx=np.nanmean(BRS_Mx)
       results.append(BRS_Mx)
    else:
       BRS_Mx=np.nan
       results.append(BRS_Mx)
    
    #append results
    results_iteration.append(results)
    
     
    
 
#recall of Function to headers with values
results_final = concatenate_list(results_iteration,dictionary_list_test)

# recall of Function to remove outliers
a = results_final.copy()
results_final= outliers(results_final, 3.0)



#########################IPSI AND CONTRA - CHANGE SIDE########################
#recall of Function to read side:               



#recall of Function to add empty(NaN) columns with 'ipsi' and 'contra' name

adding_new_ipsi_columns (ipsi_list, results_final)

#filling ipsi and contra columns in each steps:


for n in range(0,files_number):

    side_patient = side.loc[n,:].values[1]
    if side_patient == 2:
        results_final['Mx_ipsi'].values[n] = results_final['Mx_L'].values[n]
        results_final['Mx_contra'].values[n] = results_final['Mx_R'].values[n]
        results_final['FV_ipsi'].values[n] = results_final['FVL'].values[n]
        results_final['FV_contra'].values[n] = results_final['FVR'].values[n]
                    
        
    elif side_patient == 1:
        results_final['Mx_ipsi'].values[n] = results_final['Mx_R'].values[n]
        results_final['Mx_contra'].values[n] = results_final['Mx_L'].values[n]
        results_final['FV_ipsi'].values[n] = results_final['FVR'].values[n]
        results_final['FV_contra'].values[n] = results_final['FVL'].values[n]
        
        
    elif side_patient == 3:
        Mx = np.array([results_final['Mx_L'].values[n],results_final['Mx_R'].values[n]])
        
        if np.isnan(Mx).all() == True:
            results_final['Mx_ipsi'].values[n] = np.nan
            
        elif np.isnan(Mx).any() == True:
             results_final['Mx_ipsi'].values[n] = np.nansum(Mx)
        else:
            results_final['Mx_ipsi'].values[n] = np.nansum(Mx)/2
        results_final['Mx_contra'].values[n] = np.nan
        
        FV = np.array([results_final['FVL'].values[n], results_final['FVR'].values[n]])
        if np.isnan(FV).all() == True:
            results_final['FV_ipsi'].values[n] = np.nan
        elif np.isnan(FV).any() == True:
            results_final['FV_ipsi'].values[n] = np.nansum(FV)/2
        else:
             results_final['FV_ipsi'].values[n] = np.nansum(FV)/2
        results_final['FV_contra'].values[n] = np.nan
        
       
        
    else:
        results_final['Mx_ipsi'].values[n] = np.nan
        results_final['Mx_contra'].values[n] = np.nan
        results_final['FV_ipsi'].values[n] = np.nan
        results_final['FV_contra'].values[n] = np.nan
        

########################################################################3


#saving results


results_final.to_csv(SOURCE_FILE_PATH + '/' +'results.csv', sep = ';', index = False)




