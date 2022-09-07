
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\wyniki_per_patient"
SOURCE_SPASM = "C:\Moje_dokumenty\Po_doktoracie_09_2021\Tau_DCI\side_of_vasospasm.csv"
EXTENSION_FILE = ".csv"
PATIENTS = 97

dictionary_list =['Mxa_L', 'Mxa_R', 'ABP_inv', 'AmpABP_inv','FVL', 'FVR', 'AmpFVL', 'AmpFVR',\
                  'HR_inv','HR_Hz_inv', 'CaBV_L_CFF', 'CaBV_R_CFF', 'AmpCaBV_L_CFF', 'AmpCaBV_R_CFF',\
                   'CVR2L',	'CVR2R','Ca_CFF1L', 'Ca_CFF1R', 'ICP', 'Tau_CFF_L', 'Tau_CFF_R']


ipsi_list = ['FV_ipsi', 'FV_contra', 'AmpFV_ipsi', 'AmpFV_contra', \
             'CVR_ipsi', 'CVR_contra', 'Ca_CFF_ipsi', 'Ca_CFF_contra', \
             'Mxa_ipsi', 'Mxa_contra', 'T_CFF_ipsi', 'T_CFF_contra' ,\
             'CVR_average', 'Ca_CFF_average', 'T_CFF_average', 'FV_average', 'AmpFV_average', \
             'Mxa_average']
    
    
remove_list = ['FVL', 'FVR', 'AmpFVL', 'AmpFVR',\
               'CVR2L', 'CVR2R', 'Ca_CFF1L', 'Ca_CFF1R' ,\
               'Tau_CFF_R', 'Tau_CFF_L','Mxa_L', 'Mxa_R' ]


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
    dataset[dataset<= 0] = np.nan
    test_to_remove = dataset.copy() 
    ############# ############# ############# #############
   
         
    return dataset, test_to_remove


#function to find extreme values
def outliers(data, ex):
  
        
    #lim = np.logical_or(data < data.quantile(0.99, numeric_only=False),data > data.quantile(0.01, numeric_only=False))
    lim = np.abs((data - data.mean()) / data.std(ddof=0)) < 3   
    d_f = data.where(lim, np.nan)
    
                          
   
    return d_f
  
      

#Function to find NaN and replace by mean
def replace_NaN (data): 
    
    total_rows=len(data.axes[0])
    total_cols=len(data.axes[1])
    pusta_macierz = pd.DataFrame(index=np.arange(total_rows), columns=np.arange(total_cols))
    for i in range (0, total_cols):
        a = data.iloc[:,i]
             
        if a.isnull().sum() <(len(data.index)):
                      
            a = pd.DataFrame(a)
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            imputer = imputer.fit(a)
            transform_results = imputer.transform(a)
            transform_results = pd.DataFrame(transform_results)
            
            pusta_macierz.iloc[:,i]=transform_results.values
            
        
        else:
            #transform_results  = a.fillna(9999)
            transform_results=a.fillna(method='ffill', limit=1)
            transform_results = pd.DataFrame(transform_results)
            pusta_macierz.iloc[:,i]=transform_results.values
        
            
    pusta_macierz.columns = data.columns
    return pusta_macierz


#Function to find position of variables
def find_columns(df, dictionary_list):
    
    
  
    dictionary_list_position = np.empty([1,len(dictionary_list)],dtype=int)
    dictionary_list_position[:] = np.nan
    
    for counter_list in range (0, len(dictionary_list)):
       
        for counter in range (0,len(df.columns)):
            if df.columns[counter] == dictionary_list[counter_list]:
                dictionary_list_position[:,counter_list] = counter 
                
    
    return dictionary_list_position


#Function to find mean values of each signals/variables
def mean_values(data,position, dictionary_list):
    mean_parameters = data.mean(axis=0)
    mean_parameters = pd.DataFrame(mean_parameters)
    mean_parameters =  mean_parameters.T
   
    results = pd.DataFrame(index=np.arange(1), columns=np.arange(len(dictionary_list)))
    results.columns = dictionary_list
    for i in range (0,position.size):
        if position[:,i]<0:
            results.iloc[:,i] = np.nan
        else:
            a = position[:,i]
            results.iloc[:,i]= mean_parameters.iloc[:,a]
            
       
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


###################SCRIPT###############################3

     
#recall of Function to get number of files:               
files_number = counter_files(SOURCE_FILE_PATH,EXTENSION_FILE)
        

#recall of Function to get list of files:               
files_list, only_files_names = get_list_files(SOURCE_FILE_PATH,EXTENSION_FILE)

#recall of Function to get files names without extension:
list_without_ext = get_files_names_withext(only_files_names)

results_iteration = []
for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset, test_to_remove = read_file(files_list[counter_files], separator = ';', decimal_sign = ',', delete_column = 'DateTime')

    #recall of Function to change NaN in data:
    dataset_transform = replace_NaN(dataset)

    #recall of Function to find headers from dictionary:
    position = find_columns(dataset_transform, dictionary_list)

    #recall of Function to calculate mean values
    results = mean_values(dataset_transform, position, dictionary_list)
    
    #append results    
    results_iteration.append(results)
    

     
#recall of Function to headers with values
results_final = pd.concat(results_iteration)
results_final=concatenate_list(results_final,dictionary_list)

# recall of Function to remove outliers
a = results_final.copy()
results_final= outliers(results_final, 3.0)    
 


#########################IPSI AND CONTRA - CHANGE SIDE########################
#recall of Function to read side:               
side = read_file_side(SOURCE_SPASM, separator = ';')


#recall of Function to add empty(NaN) columns with 'ipsi' and 'contra' name
adding_new_ipsi_columns (ipsi_list, results_final)

#filling ipsi and contra columns in each steps:

for n in range(0,files_number):

    side_patient = side.loc[n,:].values[1]
    if side_patient == 1:
        results_final['CVR_ipsi'].values[n] = results_final['CVR2L'].values[n]
        results_final['CVR_contra'].values[n] = results_final['CVR2R'].values[n]
        results_final['Ca_CFF_ipsi'].values[n] = results_final['Ca_CFF1L'].values[n]
        results_final['Ca_CFF_contra'].values[n] = results_final['Ca_CFF1R'].values[n]
        results_final['T_CFF_ipsi'].values[n] = results_final['Tau_CFF_L'].values[n]
        results_final['T_CFF_contra'].values[n] = results_final['Tau_CFF_R'].values[n]
        results_final['FV_ipsi'].values[n] = results_final['FVL'].values[n]
        results_final['FV_contra'].values[n] = results_final['FVR'].values[n]
        results_final['AmpFV_ipsi'].values[n] = results_final['AmpFVL'].values[n]
        results_final['AmpFV_contra'].values[n] = results_final['AmpFVR'].values[n]
        results_final['Mxa_ipsi'].values[n] = results_final['Mxa_L'].values[n]
        results_final['Mxa_contra'].values[n] = results_final['Mxa_R'].values[n]     
        
        results_final['CVR_average'].values[n] =  np.nan
        results_final['Ca_CFF_average'].values[n] =  np.nan
        results_final['T_CFF_average'].values[n] =  np.nan
        results_final['FV_average'].values[n] =  np.nan
        results_final['AmpFV_average'].values[n] = np.nan
        results_final['Mxa_average'].values[n] = np.nan

        
        
    elif side_patient == 2:
        results_final['CVR_ipsi'].values[n] = results_final['CVR2R'].values[n]
        results_final['CVR_contra'].values[n] = results_final['CVR2L'].values[n]
        results_final['Ca_CFF_ipsi'].values[n] = results_final['Ca_CFF1R'].values[n]
        results_final['Ca_CFF_contra'].values[n] = results_final['Ca_CFF1L'].values[n]
        results_final['T_CFF_ipsi'].values[n] = results_final['Tau_CFF_R'].values[n]
        results_final['T_CFF_contra'].values[n] = results_final['Tau_CFF_L'].values[n]
        results_final['FV_ipsi'].values[n] = results_final['FVR'].values[n]
        results_final['FV_contra'].values[n] = results_final['FVL'].values[n]
        results_final['AmpFV_ipsi'].values[n] = results_final['AmpFVR'].values[n]
        results_final['AmpFV_contra'].values[n] = results_final['AmpFVL'].values[n]
        results_final['Mxa_ipsi'].values[n] = results_final['Mxa_R'].values[n]
        results_final['Mxa_contra'].values[n] = results_final['Mxa_L'].values[n]
        
        results_final['CVR_average'].values[n] =  np.nan
        results_final['Ca_CFF_average'].values[n] =  np.nan
        results_final['T_CFF_average'].values[n] =  np.nan
        results_final['FV_average'].values[n] =  np.nan
        results_final['AmpFV_average'].values[n] = np.nan
        results_final['Mxa_average'].values[n] = np.nan
        
        
        
        
        
        
    elif side_patient == 3:
        CVR = np.array([results_final['CVR2R'].values[n],results_final['CVR2L'].values[n]])
        if np.isnan(CVR).any() == True:
            results_final['CVR_average'].values[n] = np.nansum(CVR)
        else:
            results_final['CVR_average'].values[n] = np.nansum(CVR)/2
        results_final['CVR_contra'].values[n] = np.nan
        results_final['CVR_ipsi'].values[n] = np.nan
        
        CA = np.array([results_final['Ca_CFF1R'].values[n], results_final['Ca_CFF1L'].values[n]])
        if np.isnan(CA).any() == True:
            results_final['Ca_CFF_average'].values[n] = np.nansum(CA)
        else:
             results_final['Ca_CFF_average'].values[n] = np.nansum(CA)/2
        results_final['Ca_CFF_ipsi'].values[n] = np.nan
        results_final['Ca_CFF_contra'].values[n] = np.nan
        
     
               
        TCF =  np.array([results_final['Tau_CFF_R'].values[n], results_final['Tau_CFF_L'].values[n]])
        if np.isnan(TCF).any() == True:
            results_final['T_CFF_average'].values[n] = np.nansum(TCF)
        else:
            results_final['T_CFF_average'].values[n] = np.nansum(TCF)/2
        results_final['T_CFF_ipsi'].values[n] = np.nan
        results_final['T_CFF_contra'].values[n] = np.nan
        
        
               
        FVI = np.array([results_final['FVR'].values[n] , results_final['FVL'].values[n]])
        if np.isnan(FVI).any() == True:
            results_final['FV_average'].values[n] = np.nansum(FVI)
        else:
            results_final['FV_average'].values[n] = np.nansum(FVI)/2
        results_final['FV_ipsi'].values[n] = np.nan
        results_final['FV_contra'].values[n] = np.nan
        
        FVC = np.array([results_final['AmpFVR'].values[n] , results_final['AmpFVL'].values[n]])
        if np.isnan(FVC).any() == True:
            results_final['AmpFV_average'].values[n] = np.nansum(FVC)
        else:
            results_final['AmpFV_average'].values[n] = np.nansum(FVC)/2
        results_final['AmpFV_ipsi'].values[n] = np.nan
        results_final['AmpFV_contra'].values[n] = np.nan
        
        MXA = np.array([results_final['Mxa_R'].values[n] , results_final['Mxa_L'].values[n]])
        if np.isnan(MXA).any() == True:
            results_final['Mxa_average'].values[n] = np.nansum(MXA)
        else:
            results_final['Mxa_average'].values[n] = np.nansum(MXA)/2
        results_final['Mxa_ipsi'].values[n] = np.nan
        results_final['Mxa_contra'].values[n] = np.nan
        
        
    else:
        results_final['CVR_ipsi'].values[n] = np.nan
        results_final['CVR_contra'].values[n] = np.nan
        results_final['Ca_CFF_ipsi'].values[n] = np.nan
        results_final['Ca_CFF_contra'].values[n] = np.nan
        results_final['T_CFF_ipsi'].values[n] = np.nan
        results_final['T_CFF_contra'].values[n] = np.nan
        results_final['FV_ipsi'].values[n] = np.nan
        results_final['FV_contra'].values[n] = np.nan
        results_final['AmpFV_ipsi'].values[n] = np.nan
        results_final['AmpFV_contra'].values[n] = np.nan
        results_final['Mxa_ipsi'].values[n] = np.nan
        results_final['Mxa_contra'].values[n] = np.nan


########################################################################3


#create empty matrix with columns names the same as in results_final and putting results in place the same as patients
pusta_macierz = pd.DataFrame(index=np.arange(PATIENTS+1), columns=np.arange(len(results_final.columns)))
for n in range (0,files_number):
    pusta_macierz.values[list_without_ext[n]] = results_final.values[n]
pusta_macierz.columns = results_final.columns


####remove left and right columns
remove_LR_columns (remove_list,pusta_macierz)
pusta_macierz = pusta_macierz.drop(pusta_macierz.index[0])

#saving results
pusta_macierz.to_csv(SOURCE_FILE_PATH + '/' +'results.csv', sep = ';', index = False)






