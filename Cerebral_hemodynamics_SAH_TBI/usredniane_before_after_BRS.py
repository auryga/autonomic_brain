# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:29:45 2020

@author: agnie
"""

import pandas as pd
import numpy as np



PATH = "C:\Moje_dokumenty\Po_doktoracie\TAU_DCI_SKURCZ\BRS\opis\dane_BRS.csv"
PATH2 = "C:\\Moje_dokumenty\\Po_doktoracie\\TAU_DCI_SKURCZ\\BRS\\opis\\results_total.csv"
SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie\TAU_DCI_SKURCZ\BRS\opis"

# Function to read file
def read_file(file_name, separator = None, decimal_sign = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    return dataset



info = read_file(PATH, separator = ';', decimal_sign = '.')
dataset = read_file(PATH2, separator = ';', decimal_sign = '.')

dataset = pd.concat([info, dataset], axis=1)


# for i in range (348,471):
#    dataset.drop(index=[i], inplace = true)
        
sredni_brs = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_hf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_hf_r=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_hf_n=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_lf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_lf_r=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_lf_n=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_lf_hf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_tp=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hr_sdsd=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_hrv_rmssd=pd.DataFrame(index=np.arange(97), columns=np.arange(1))    
sredni_abp=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_mx_ipsi=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
sredni_mx_contra=pd.DataFrame(index=np.arange(97), columns=np.arange(1))                    
sredni_fv_ipsi=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
sredni_fv_contra=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
sredni_BRSMx=pd.DataFrame(index=np.arange(97), columns=np.arange(1))                           


before_brs = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_hf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_hf_r=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_hf_n=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_lf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_lf_r=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_lf_n=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_lf_hf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_tp=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hr_sdsd=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_hrv_rmssd=pd.DataFrame(index=np.arange(97), columns=np.arange(1))    
before_abp=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_mx_ipsi=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
before_mx_contra=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
before_fv_ipsi=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
before_fv_contra=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_BRSMx=pd.DataFrame(index=np.arange(97), columns=np.arange(1))  

during_brs = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_hf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_hf_r=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_hf_n=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_lf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_lf_r=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_lf_n=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_lf_hf=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_tp=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hr_sdsd=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_hrv_rmssd=pd.DataFrame(index=np.arange(97), columns=np.arange(1))    
during_abp=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_mx_ipsi=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_mx_contra=pd.DataFrame(index=np.arange(97), columns=np.arange(1))  
during_fv_ipsi=pd.DataFrame(index=np.arange(97), columns=np.arange(1)) 
during_fv_contra=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_BRSMx=pd.DataFrame(index=np.arange(97), columns=np.arange(1))  

cv=pd.DataFrame(index=np.arange(97), columns=np.arange(1))

    
    
for n in range (1,98):
     old = dataset.NR == n
     old2 = dataset.CV[old]
     if sum(old2) == 0:
        sredni_brs.iloc[n-1] =  np.nanmean(dataset.BRS[old])
        sredni_hr.iloc[n-1] = np.nanmean(dataset.HR[old])
        sredni_hr_hf.iloc[n-1] = np.nanmean(dataset.HRV_HF[old])
        sredni_hr_hf_r.iloc[n-1] = np.nanmean(dataset.HRhfRel[old])
        sredni_hr_hf_n.iloc[n-1] = np.nanmean(dataset.HRhfN[old])
        sredni_hr_lf.iloc[n-1] = np.nanmean(dataset.HRV_LF[old])
        sredni_hr_lf_r.iloc[n-1] = np.nanmean(dataset.HRlfDivhtRel[old])
        sredni_hr_lf_n.iloc[n-1] = np.nanmean(dataset.HRlfDivhtNorm[old])
        sredni_hr_lf_hf.iloc[n-1] = np.nanmean(dataset.HRV_LFHF[old])
        sredni_hr_tp.iloc[n-1] = np.nanmean(dataset.HRV_TP[old])
        sredni_hr_sdsd.iloc[n-1] = np.nanmean(dataset.HRV_SDSD[old])
        sredni_hrv_rmssd.iloc[n-1] = np.nanmean(dataset.HRV_RMSSD[old])
        sredni_abp.iloc[n-1] = np.nanmean(dataset.abp[old])
        sredni_mx_ipsi.iloc[n-1] = np.nanmean(dataset.Mx_ipsi[old])
        sredni_mx_contra.iloc[n-1] =  np.nanmean(dataset.Mx_contra[old])                  
        sredni_fv_ipsi.iloc[n-1] = np.nanmean(dataset.FV_ipsi[old])
        sredni_fv_contra.iloc[n-1] = np.nanmean(dataset.FV_contra[old])
        sredni_BRSMx.iloc[n-1]=np.nanmean(dataset.BRS_Mx[old]) 
        cv.iloc[n-1]=min(dataset.CV[old])
     else:
        sredni_brs.iloc[n-1] =  np.nan
        sredni_hr.iloc[n-1] = np.nan
        sredni_hr_hf.iloc[n-1] = np.nan
        sredni_hr_hf_r.iloc[n-1] = np.nan
        sredni_hr_hf_n.iloc[n-1] = np.nan
        sredni_hr_lf.iloc[n-1] = np.nan
        sredni_hr_lf_r.iloc[n-1] = np.nan
        sredni_hr_lf_n.iloc[n-1] = np.nan
        sredni_hr_lf_hf.iloc[n-1] = np.nan
        sredni_hr_tp.iloc[n-1] = np.nan
        sredni_hr_sdsd.iloc[n-1] = np.nan
        sredni_hrv_rmssd.iloc[n-1] = np.nan
        sredni_abp.iloc[n-1] = np.nan
        sredni_mx_ipsi.iloc[n-1] = np.nan
        sredni_mx_contra.iloc[n-1] = np.nan                   
        sredni_fv_ipsi.iloc[n-1] = np.nan
        sredni_fv_contra.iloc[n-1] = np.nan
        sredni_BRSMx.iloc[n-1]=np.nan
        
                
        before = (dataset.NR == n) & (dataset.DURING == 0)
        
        before_brs.iloc[n-1] =  np.nanmean(dataset.BRS[before])
        before_hr.iloc[n-1] = np.nanmean(dataset.HR[before])
        before_hr_hf.iloc[n-1] = np.nanmean(dataset.HRV_HF[before])
        before_hr_hf_r.iloc[n-1] = np.nanmean(dataset.HRhfRel[before])
        before_hr_hf_n.iloc[n-1] = np.nanmean(dataset.HRhfN[before])
        before_hr_lf.iloc[n-1] = np.nanmean(dataset.HRV_LF[before])
        before_hr_lf_r.iloc[n-1] = np.nanmean(dataset.HRlfDivhtRel[before])
        before_hr_lf_n.iloc[n-1] = np.nanmean(dataset.HRlfDivhtNorm[before])
        before_hr_lf_hf.iloc[n-1] = np.nanmean(dataset.HRV_LFHF[before])
        before_hr_tp.iloc[n-1] = np.nanmean(dataset.HRV_TP[before])
        before_hr_sdsd.iloc[n-1] = np.nanmean(dataset.HRV_SDSD[before])
        before_hrv_rmssd.iloc[n-1] = np.nanmean(dataset.HRV_RMSSD[before])
        before_abp.iloc[n-1] = np.nanmean(dataset.abp[before])
        before_mx_ipsi.iloc[n-1] = np.nanmean(dataset.Mx_ipsi[before])
        before_mx_contra.iloc[n-1] =  np.nanmean(dataset.Mx_contra[before])                  
        before_fv_ipsi.iloc[n-1] = np.nanmean(dataset.FV_ipsi[before])
        before_fv_contra.iloc[n-1] = np.nanmean(dataset.FV_contra[before])
        before_BRSMx.iloc[n-1]=np.nanmean(dataset.BRS_Mx[before]) 
                  
        
        during = (dataset.NR == n) & (dataset.DURING == 1)
        
        during_brs.iloc[n-1] =  np.nanmean(dataset.BRS[during])
        during_hr.iloc[n-1] = np.nanmean(dataset.HR[during])
        during_hr_hf.iloc[n-1] = np.nanmean(dataset.HRV_HF[during])
        during_hr_hf_r.iloc[n-1] = np.nanmean(dataset.HRhfRel[during])
        during_hr_hf_n.iloc[n-1] = np.nanmean(dataset.HRhfN[during])
        during_hr_lf.iloc[n-1] = np.nanmean(dataset.HRV_LF[during])
        during_hr_lf_r.iloc[n-1] = np.nanmean(dataset.HRlfDivhtRel[during])
        during_hr_lf_n.iloc[n-1] = np.nanmean(dataset.HRlfDivhtNorm[during])
        during_hr_lf_hf.iloc[n-1] = np.nanmean(dataset.HRV_LFHF[during])
        during_hr_tp.iloc[n-1] = np.nanmean(dataset.HRV_TP[during])
        during_hr_sdsd.iloc[n-1] = np.nanmean(dataset.HRV_SDSD[during])
        during_hrv_rmssd.iloc[n-1] = np.nanmean(dataset.HRV_RMSSD[during])
        during_abp.iloc[n-1] = np.nanmean(dataset.abp[during])
        during_mx_ipsi.iloc[n-1] = np.nanmean(dataset.Mx_ipsi[during])
        during_mx_contra.iloc[n-1] =  np.nanmean(dataset.Mx_contra[during])                  
        during_fv_ipsi.iloc[n-1] = np.nanmean(dataset.FV_ipsi[during])
        during_fv_contra.iloc[n-1] = np.nanmean(dataset.FV_contra[during])
        during_BRSMx.iloc[n-1]=np.nanmean(dataset.BRS_Mx[during]) 


        
        cv.iloc[n-1]=max(dataset.CV[old])
        
        
        
     n = n+1    
    
results_final= pd.DataFrame(index=np.arange(97), columns=np.arange(18))

frames = [sredni_brs, sredni_hr, sredni_hr_hf, sredni_hr_hf_r, sredni_hr_hf_n, 
          sredni_hr_lf, sredni_hr_lf_r, sredni_hr_lf_n,sredni_hr_lf_hf, sredni_hr_tp, 
          sredni_hr_sdsd, sredni_hrv_rmssd, sredni_abp,
          sredni_mx_ipsi, sredni_mx_contra, sredni_fv_ipsi, sredni_fv_contra,sredni_BRSMx,
          before_brs, before_hr, before_hr_hf, before_hr_hf_r, before_hr_hf_n, 
          before_hr_lf, before_hr_lf_r,before_hr_lf_n,before_hr_lf_hf, before_hr_tp, 
          before_hr_sdsd, before_hrv_rmssd, before_abp,before_mx_ipsi, before_mx_contra, 
          before_fv_ipsi, before_fv_contra, before_BRSMx,during_brs, during_hr, during_hr_hf, during_hr_hf_r, 
          during_hr_hf_n, during_hr_lf, during_hr_lf_r, during_hr_lf_n,during_hr_lf_hf, 
          during_hr_tp, during_hr_sdsd, during_hrv_rmssd, during_abp,
          during_mx_ipsi, during_mx_contra, during_fv_ipsi, during_fv_contra, during_BRSMx,cv]

results_final=pd.concat(frames, axis=1)
results_final.columns=(['sredni_brs', 'sredni_hr', 'sredni_hr_hf', 'sredni_hr_hf_r',
'sredni_hr_hf_n','sredni_hr_lf', 'sredni_hr_lf_r', 'sredni_hr_lf_n','sredni_hr_lf_hf', 
'sredni_hr_tp','sredni_hr_sdsd', 'sredni_hrv_rmssd', 'sredni_abp','sredni_mx_ipsi', 
'sredni_mx_contra', 'sredni_fv_ipsi','sredni_fv_contra','sredni_BRSMx','before_brs', 'before_hr', 'before_hr_hf', 
'before_hr_hf_r','before_hr_hf_n','before_hr_lf', 'before_hr_lf_r','before_hr_lf_n','before_hr_lf_hf', 
'before_hr_tp','before_hr_sdsd', 'before_hrv_rmssd', 'before_abp','before_mx_ipsi', 'before_mx_contra',
'before_fv_ipsi', 'before_fv_contra', 'before_BRSMx','during_brs', 'during_hr', 'during_hr_hf', 'during_hr_hf_r', 
'during_hr_hf_n', 'during_hr_lf', 'during_hr_lf_r', 'during_hr_lf_n','during_hr_lf_hf','during_hr_tp',
 'during_hr_sdsd', 'during_hrv_rmssd', 'during_abp','during_mx_ipsi', 'during_mx_contra', 
 'during_fv_ipsi', 'during_fv_contra','during_BRSMx','CV'])
    
results_final.to_csv(SOURCE_FILE_PATH + '/' +'results_before_during_BRS.csv', sep = ';', index = False)
    

