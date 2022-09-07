
import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import math
import statistics as st
import matplotlib.pyplot as plt

SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_06_2019_08_2021\zaproszony_artykul\summary"
SOURCE_SPASM = "C:\Moje_dokumenty\Po_doktoracie_06_2019_08_2021\zaproszony_artykul\strony\side.csv"
EXTENSION_FILE = ".csv"
SAVE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie_06_2019_08_2021\zaproszony_artykul"


dictionary_list =['BRS', 'HR', 'abp', 
                  'CI', 'CO', 'STO2LEFT', 'STO2RIGHT', 'Tox_R','TOx_L']


   
ipsi_list = lista=['rso2_ipsi_during','abp_ipsi_during', 'HR_ipsi_during', 'TOX_ipsi_during',
    'co_ipsi_during', 'ci_ipsi_during', 'brs_ipsi_during', 'ipsi_ilosc_spadkow','ipsi_indeksy',
    'rso2_ipsi_before','abp_ipsi_before', 'HR_ipsi_before', 'TOX_ipsi_before',
    'co_ipsi_before', 'ci_ipsi_before', 'brs_ipsi_before', 'rso2_ipsi_length','indeksy_length_ipsi','rso2_contra_during','abp_contra_during', 'HR_contra_during', 'TOX_contra_during',
    'co_contra_during', 'ci_contra_during', 'brs_contra_during', 'contra_ilosc_spadkow','contra_indeksy',
    'rso2_contra_before','abp_contra_before', 'HR_contra_before', 'TOX_contra_before',
    'co_contra_before', 'ci_contra_before', 'brs_contra_before','rso2_contra_length','indeksy_length_contra'] 
    
               


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
        df_out = kolumna.loc[(kolumna < fence_low )| (kolumna > fence_high)]
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
        #print(i)
        if math.isnan(position[0,i])==True:
            #print('nan')
            d=np.nan
        elif data.empty == True:
            d=np.nan
        else:
            #print(i)
            d=data.iloc[1, int(position[0,i])].mean()
            
            
            
        results.append(d)
    
    return results

#Function to cut signals into 5-minutes windows:
def windows(data,dictionary_list,position,maska):
    ilosc_spadkow=0
    rso2_left_during =[]
    abp_during=[]
    HR_during=[]
    TOX_left_during=[]
    co_during=[]
    ci_during=[]
    brs_during=[]
   
    
    rso2_left_before =[]
    abp_before=[]
    HR_before=[]
    TOX_left_before=[]
    co_before=[]
    ci_before=[]
    brs_before=[]
    
    
    indeksy =[]
    time = len(data.index)
    
    rso2_left=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))  
    abp=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))  
    HR=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))  
    TOX_l=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))   
    co=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))
    ci=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1)) 
    brs=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))
   
   
   
   
    
    
    if maska.iloc[0,5]==1:
        rso2_left=data.iloc[:,position[:,5]]
        rso2_left=rso2_left.apply(lambda x: np.where(x < 45,np.nan,x))
    if maska.iloc[0,2]==1:
        abp=data.iloc[:,position[:,2]]
        abp=abp.apply(lambda x: np.where(x < 20,np.nan,x))
    if maska.iloc[0,1]==1:
        HR=data.iloc[:,position[:,1]]
        HR=HR.apply(lambda x: np.where(x < 20,np.nan,x))
    if maska.iloc[0,8]==1:
        TOX_l=data.iloc[:,position[:,8]]
        
    if maska.iloc[0,4]==1:
        co=data.iloc[:,position[:,4]]
        co=co.apply(lambda x: np.where(x < 0.5,np.nan,x))
    if maska.iloc[0,3]==1:
        ci=data.iloc[:,position[:,3]]
        ci=ci.apply(lambda x: np.where(x < 0.5,np.nan,x))
    if maska.iloc[0,0]==1:
        brs=data.iloc[:,position[:,0]]
        brs=brs.apply(lambda x: np.where(x > 35,np.nan,x))
        brs=brs.apply(lambda x: np.where(x < 1,np.nan,x))
    
    window=30
    #okno 5 minutowe   
    if maska.iloc[0,5]==1:
        for j in range(0,len(rso2_left),5):
            mean_window=np.nanmean(rso2_left[j:j+window])
            ignore=(rso2_left[j:j+window].count())
            
            if (mean_window<=60) & (ignore[0]>(0.8*window)):
            
                ilosc_spadkow+=1
                rso2_left_during.append(np.nanmean(rso2_left[j:j+window]))
                HR_during.append(np.nanmean(HR[j:j+window]))
                abp_during.append(np.nanmean(abp[j:j+window]))
                TOX_left_during.append(np.nanmean(TOX_l[j:j+window]))
                co_during.append(np.nanmean(co[j:j+window]))
                ci_during.append(np.nanmean(ci[j:j+window]))
                brs_during.append(np.nanmean(brs[j:j+window]))
               
                
                if j+window<len(rso2_left):
                    for i in range(j,j+window+1):
                        indeksy.append(i)
        
        indeksy=set(indeksy)
        indeksy=sorted(list(indeksy))
       
        indeksy_lenght_left=len(indeksy)   
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
        rso2_left_length=int((rso2_left.count())/60)

        
        
    results_during_before= pd.DataFrame(index=np.arange(1), columns=np.arange(19))

    frames = [rso2_left_during,abp_during, HR_during, TOX_left_during,
    co_during, ci_during, brs_during, ilosc_spadkow,indeksy,
    rso2_left_before,abp_before, HR_before, TOX_left_before,
    co_before, ci_before, brs_before,rso2_left_length, indeksy_lenght_left, time]

        
    results_during_before.loc[len(results_during_before)] = frames
    
    results_during_before.columns=(['rso2_left_during','abp_during', 'HR_during', 'TOX_left_during',
    'co_during', 'ci_during', 'brs_during', 'ilosc_spadkow','indeksy',
    'rso2_left_before','abp_before', 'HR_before', 'TOX_left_before',
    'co_before', 'ci_before', 'brs_before', 'rso2_left_length','indeksy_lenght_left','time'])
                
            
    return results_during_before
    

#Function to cut signals into 5-minutes windows:
def windows_right(data,dictionary_list,position,maska):
    ilosc_spadkow_right=0
    rso2_right_during =[]
    abp_during_right=[]
    HR_during_right=[]
    TOX_right_during=[]
    co_during_right=[]
    ci_during_right=[]
    brs_during_right=[]
    
    
    rso2_right_before =[]
    abp_before_right=[]
    HR_before_right=[]
    TOX_right_before=[]
    co_before_right=[]
    ci_before_right=[]
    brs_before_right=[]
    

    
    indeksy_right =[]
    
    rso2_right=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))  
    abp=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))  
    HR=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))  
    TOX_r=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))   
    co=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))
    ci=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1)) 
    brs=pd.DataFrame(0,index=np.arange(len(data.index)), columns=np.arange(1))
    
    
    if maska.iloc[0,6]==1:
        rso2_right=data.iloc[:,position[:,6]]
        rso2_right=rso2_right.apply(lambda x: np.where(x < 45,np.nan,x))
    if maska.iloc[0,2]==1:
        abp=data.iloc[:,position[:,2]]
        abp=abp.apply(lambda x: np.where(x < 20,np.nan,x))
    if maska.iloc[0,1]==1:
        HR=data.iloc[:,position[:,1]]
        HR=HR.apply(lambda x: np.where(x < 20,np.nan,x))
    if maska.iloc[0,7]==1:
        TOX_r=data.iloc[:,position[:,7]]
    if maska.iloc[0,4]==1:
        co=data.iloc[:,position[:,4]]
        co=co.apply(lambda x: np.where(x < 0.5,np.nan,x))
    if maska.iloc[0,3]==1:
        ci=data.iloc[:,position[:,3]]
        ci=ci.apply(lambda x: np.where(x < 0.5,np.nan,x))
    if maska.iloc[0,0]==1:
        brs=data.iloc[:,position[:,0]] 
        brs=brs.apply(lambda x: np.where(x > 35,np.nan,x))
        brs=brs.apply(lambda x: np.where(x < 1,np.nan,x))
    
     
    window=30
    #okno 5 minutowe   
    
    if position[:,6]!=np.nan:  
        for j in range(0,len(rso2_right),5):
            mean_window=np.nanmean(rso2_right[j:j+window])
            ignore=(rso2_right[j:j+window].count())
            
            if (mean_window<=60) & (ignore[0]>(0.8*window)):
                ilosc_spadkow_right+=1
                rso2_right_during.append(np.nanmean(rso2_right[j:j+window]))
                HR_during_right.append(np.nanmean(HR[j:j+window]))
                abp_during_right.append(np.nanmean(abp[j:j+window]))
                TOX_right_during.append(np.nanmean(TOX_r[j:j+window]))
                co_during_right.append(np.nanmean(co[j:j+window]))
                ci_during_right.append(np.nanmean(ci[j:j+window]))
                brs_during_right.append(np.nanmean(brs[j:j+window]))
                
                
                
                if j+window<len(rso2_right):
                    for i in range(j,j+window+1):
                        indeksy_right.append(i)
                    
        
        
        
        
        
        
        maska=pd.DataFrame(index=np.arange(len(rso2_right)), columns=np.arange(1))
        indeksy_right=set(indeksy_right)
        indeksy_lenght_right=len(indeksy_right)  
          
        maska=maska.replace(np.nan, 1)
        for i in indeksy_right:
            maska.iloc[i,:]=np.nan
        
                
           
        rso2_right_during = np.nanmean(rso2_right_during)
        abp_during_right= np.nanmean(abp_during_right)
        HR_during_right=np.nanmean(HR_during_right)
        TOX_right_during=np.nanmean(TOX_right_during)
        co_during_right=np. nanmean(co_during_right)
        ci_during_right=np.nanmean(ci_during_right)
        brs_during_right=np.nanmean(brs_during_right)
        
        
        
        rso2_right_before = np.nanmean(rso2_right.iloc[:,0]*maska.iloc[:,0])
        abp_before_right= np.nanmean(abp.iloc[:,0]*maska.iloc[:,0])
        HR_before_right=np.nanmean(HR.iloc[:,0]*maska.iloc[:,0])
        TOX_right_before=np.nanmean(TOX_r.iloc[:,0]*maska.iloc[:,0])
        co_before_right=np.nanmean(co.iloc[:,0]*maska.iloc[:,0])
        ci_before_right=np.nanmean(ci.iloc[:,0]*maska.iloc[:,0])
        brs_before_right=np.nanmean(brs.iloc[:,0]*maska.iloc[:,0])
        rso2_right_length=int((rso2_right.count())/60)
        
        
    results_during_before= pd.DataFrame(index=np.arange(1), columns=np.arange(18))

    frames = [rso2_right_during,abp_during_right, HR_during_right, TOX_right_during,
    co_during_right, ci_during_right, brs_during_right, ilosc_spadkow_right,indeksy_right,
    rso2_right_before,abp_before_right, HR_before_right, TOX_right_before,
    co_before_right, ci_before_right, brs_before_right, rso2_right_length,indeksy_lenght_right]

        
    results_during_before.loc[len(results_during_before)] = frames
    
    results_during_before.columns=(['rso2_right_during','abp_during', 'HR_during', 'TOX_right_during',
    'co_during', 'ci_during', 'brs_during', 'ilosc_spadkow','indeksy',
    'rso2_right__before','abp_before', 'HR_before', 'TOX_right_before',
    'co_before', 'ci_before', 'brs_before','rso2_right_length','indeksy_lenght_right'])
                
            
    return results_during_before

  

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


results_iteration = []
for counter_files in range(0,files_number):

    #recall of Function to read File:               
    dataset, test_to_remove = read_file(files_list[counter_files], separator = ';', decimal_sign = ',', delete_column = 'DateTime')

    
    
    #dataset.abp[(dataset['abp'] < 50) | (dataset['abp'] > 250) ] = np.nan
    #dataset.STO2LEFT[(dataset['STO2LEFT'] < 10) | (dataset['STO2LEFT'] > 100) ]=np.nan
    #dataset.STO2RIGHT[(dataset['STO2RIGHT'] < 10) | (dataset['STO2RIGHT'] > 100) ]=np.nan
    
    
    #recall of Function to change NaN in data:
    #dataset_transform = replace_NaN(dataset)
    

    #recall of Function to find headers from dictionary:
    position,maska = find_columns(dataset, dictionary_list)
    
    r=windows(dataset, dictionary_list, position,maska)
    
    ra=windows_right(dataset, dictionary_list, position,maska)
    frames = [r,ra]

    result = pd.concat(frames, axis = 1) 
    result = result.drop(labels=0, axis=0)
    results_iteration.append(result)
    
  
    

    
    
    
    

    
lista=['rso2_left_during','abp_left_during', 'HR_left_during', 'TOX_left_during',
    'co_left_during', 'ci_left_during', 'brs_left_during', 'ilosc_spadkow_left','indeksy_left',
    'rso2_left_before','abp_left_before', 'HR_left_before', 'TOX_left_before',
    'co_left_before', 'ci_left_before', 'brs_left_before', 'rso2_left_length','indeksy_lenght_left','time','rso2_right_during','abp_right_during', 'HR_right_during', 'TOX_right_during',
    'co_right_during', 'ci_right_during', 'brs_right_during', 'ilosc_spadkow_right','indeksy_right',
    'rso2_right_before','abp_right_before', 'HR_right_before', 'TOX_right_before',
    'co_right_before', 'ci_right_before', 'brs_right_before','rso2_right_length','indeksy_lenght_right']  
results_iteration = pd.concat(results_iteration)

results_iteration.columns = lista

#size = 32
#for i in range (0,size):
                                                                     

#results_iteration.rename(columns={results_iteration.columns[i]: lista[i] for i in range(size)}, inplace = True)





adding_new_ipsi_columns (ipsi_list, results_iteration)

for n in range(0,files_number):

    side_patient = side.loc[n,:].values[0]
    if side_patient == 'L':
        results_iteration['rso2_ipsi_during'].values[n] = results_iteration['rso2_left_during'].values[n]
        results_iteration['abp_ipsi_during'].values[n] = results_iteration['abp_left_during'].values[n]
        results_iteration['HR_ipsi_during'].values[n] = results_iteration['HR_left_during'].values[n]
        results_iteration['TOX_ipsi_during'].values[n] = results_iteration['TOX_left_during'].values[n]
        results_iteration['co_ipsi_during'].values[n] = results_iteration['co_left_during'].values[n]
        results_iteration['ci_ipsi_during'].values[n] = results_iteration['ci_left_during'].values[n]
        results_iteration['brs_ipsi_during'].values[n] = results_iteration['brs_left_during'].values[n]
        results_iteration['ipsi_ilosc_spadkow'].values[n] = results_iteration['ilosc_spadkow_left'].values[n]
        #results_iteration['ipsi_indeksy'].values[n] = results_iteration['indeksy_left'].values[n]
        results_iteration['rso2_ipsi_before'].values[n] = results_iteration['rso2_left_before'].values[n]
        results_iteration['abp_ipsi_before'].values[n] = results_iteration['abp_left_before'].values[n]
        results_iteration['HR_ipsi_before'].values[n] = results_iteration['HR_left_before'].values[n]
        results_iteration['TOX_ipsi_before'].values[n] = results_iteration['TOX_left_before'].values[n]
        results_iteration['co_ipsi_before'].values[n] = results_iteration['co_left_before'].values[n]
        results_iteration['ci_ipsi_before'].values[n] = results_iteration['ci_left_before'].values[n]
        results_iteration['brs_ipsi_before'].values[n] = results_iteration['brs_left_before'].values[n]
        results_iteration['rso2_ipsi_length'].values[n] = results_iteration['rso2_left_length'].values[n]
        results_iteration['indeksy_length_ipsi'].values[n] = results_iteration['indeksy_lenght_left'].values[n]
        
        results_iteration['rso2_contra_during'].values[n] = results_iteration['rso2_right_during'].values[n]
        results_iteration['abp_contra_during'].values[n] = results_iteration['abp_right_during'].values[n]
        results_iteration['HR_contra_during'].values[n] = results_iteration['HR_right_during'].values[n]
        results_iteration['TOX_contra_during'].values[n] = results_iteration['TOX_right_during'].values[n]
        results_iteration['co_contra_during'].values[n] = results_iteration['co_right_during'].values[n]
        results_iteration['ci_contra_during'].values[n] = results_iteration['ci_right_during'].values[n]
        results_iteration['brs_contra_during'].values[n] = results_iteration['brs_right_during'].values[n]
        results_iteration['contra_ilosc_spadkow'].values[n] = results_iteration['ilosc_spadkow_right'].values[n]
        #results_iteration['contra_indeksy'].values[n] = results_iteration['indeksy_right'].values[n]
        results_iteration['rso2_contra_before'].values[n] = results_iteration['rso2_right_before'].values[n]
        results_iteration['abp_contra_before'].values[n] = results_iteration['abp_right_before'].values[n]
        results_iteration['HR_contra_before'].values[n] = results_iteration['HR_right_before'].values[n]
        results_iteration['TOX_contra_before'].values[n] = results_iteration['TOX_right_before'].values[n]
        results_iteration['co_contra_before'].values[n] = results_iteration['co_right_before'].values[n]
        results_iteration['ci_contra_before'].values[n] = results_iteration['ci_right_before'].values[n]
        results_iteration['brs_contra_before'].values[n] = results_iteration['brs_right_before'].values[n]
        results_iteration['rso2_contra_length'].values[n] = results_iteration['rso2_right_length'].values[n]
        results_iteration['indeksy_length_contra'].values[n] = results_iteration['indeksy_lenght_right'].values[n]
    
    else:
        
        results_iteration['rso2_ipsi_during'].values[n] = results_iteration['rso2_right_during'].values[n]
        results_iteration['abp_ipsi_during'].values[n] = results_iteration['abp_right_during'].values[n]
        results_iteration['HR_ipsi_during'].values[n] = results_iteration['HR_right_during'].values[n]
        results_iteration['TOX_ipsi_during'].values[n] = results_iteration['TOX_right_during'].values[n]
        results_iteration['co_ipsi_during'].values[n] = results_iteration['co_right_during'].values[n]
        results_iteration['ci_ipsi_during'].values[n] = results_iteration['ci_right_during'].values[n]
        results_iteration['brs_ipsi_during'].values[n] = results_iteration['brs_right_during'].values[n]
        results_iteration['ipsi_ilosc_spadkow'].values[n] = results_iteration['ilosc_spadkow_right'].values[n]
        #results_iteration['ipsi_indeksy'].values[n] = results_iteration['indeksy_right'].values[n]
        results_iteration['rso2_ipsi_before'].values[n] = results_iteration['rso2_right_before'].values[n]
        results_iteration['abp_ipsi_before'].values[n] = results_iteration['abp_right_before'].values[n]
        results_iteration['HR_ipsi_before'].values[n] = results_iteration['HR_right_before'].values[n]
        results_iteration['TOX_ipsi_before'].values[n] = results_iteration['TOX_right_before'].values[n]
        results_iteration['co_ipsi_before'].values[n] = results_iteration['co_right_before'].values[n]
        results_iteration['ci_ipsi_before'].values[n] = results_iteration['ci_right_before'].values[n]
        results_iteration['brs_ipsi_before'].values[n] = results_iteration['brs_right_before'].values[n]
        results_iteration['rso2_ipsi_length'].values[n] = results_iteration['rso2_right_length'].values[n]
        results_iteration['indeksy_length_ipsi'].values[n] = results_iteration['indeksy_lenght_right'].values[n]
        
        results_iteration['rso2_contra_during'].values[n] = results_iteration['rso2_left_during'].values[n]
        results_iteration['abp_contra_during'].values[n] = results_iteration['abp_left_during'].values[n]
        results_iteration['HR_contra_during'].values[n] = results_iteration['HR_left_during'].values[n]
        results_iteration['TOX_contra_during'].values[n] = results_iteration['TOX_left_during'].values[n]
        results_iteration['co_contra_during'].values[n] = results_iteration['co_left_during'].values[n]
        results_iteration['ci_contra_during'].values[n] = results_iteration['ci_left_during'].values[n]
        results_iteration['brs_contra_during'].values[n] = results_iteration['brs_left_during'].values[n]
        results_iteration['contra_ilosc_spadkow'].values[n] = results_iteration['ilosc_spadkow_left'].values[n]
        #results_iteration['contra_indeksy'].values[n] = results_iteration['indeksy_left'].values[n]
        results_iteration['rso2_contra_before'].values[n] = results_iteration['rso2_left_before'].values[n]
        results_iteration['abp_contra_before'].values[n] = results_iteration['abp_left_before'].values[n]
        results_iteration['HR_contra_before'].values[n] = results_iteration['HR_left_before'].values[n]
        results_iteration['TOX_contra_before'].values[n] = results_iteration['TOX_left_before'].values[n]
        results_iteration['co_contra_before'].values[n] = results_iteration['co_left_before'].values[n]
        results_iteration['ci_contra_before'].values[n] = results_iteration['ci_left_before'].values[n]
        results_iteration['brs_contra_before'].values[n] = results_iteration['brs_left_before'].values[n]
        results_iteration['rso2_contra_length'].values[n] = results_iteration['rso2_left_length'].values[n]
        results_iteration['indeksy_length_contra'].values[n] = results_iteration['indeksy_lenght_left'].values[n]

number_patient = []
for n in range(0,files_number):
    number_patient.append(side.loc[n,:].values[1])
   
    
results_iteration.insert(72,'number',number_patient)


results_iteration.to_csv(SAVE_FILE_PATH + '/' +'spadki.csv', sep = ';', index = False)


