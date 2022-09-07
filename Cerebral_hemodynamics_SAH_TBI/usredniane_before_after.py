# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:29:45 2020

@author: agnie
"""

import pandas as pd
import numpy as np

PATH = "C:\Moje_dokumenty\Po_doktoracie\TAU_DCI_SKURCZ\GOTOWE\dane.csv"
SOURCE_FILE_PATH = "C:\Moje_dokumenty\Po_doktoracie\TAU_DCI_SKURCZ"


# Function to read file
def read_file(file_name, separator = None, decimal_sign = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    return dataset
    
dataset = read_file(PATH, separator = ';', decimal_sign = ',')

# for i in range (348,471):
#     dataset.drop(index=[i], inplace = True)
        
sredni_tauCFF_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_tauCFF_contra = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_tauPFF_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_tauPFF_contra = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_FV_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
sredni_FV_contra= pd.DataFrame(index=np.arange(97), columns=np.arange(1))

before_tauCFF_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_tauCFF_contra = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_tauPFF_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_tauPFF_contra = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_FV_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
before_FV_contra= pd.DataFrame(index=np.arange(97), columns=np.arange(1))

during_tauCFF_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_tauCFF_contra = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_tauPFF_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_tauPFF_contra = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_FV_ipsi = pd.DataFrame(index=np.arange(97), columns=np.arange(1))
during_FV_contra= pd.DataFrame(index=np.arange(97), columns=np.arange(1))

CV=pd.DataFrame(index=np.arange(97), columns=np.arange(1))
    
    
for n in range (1,98):
     old = dataset.NR == n
     old2 = dataset.CV[old]
     if sum(old2) == 0:
        sredni_tauCFF_ipsi.iloc[n-1] = np.nanmean(dataset.TCFF_IPSI[old])
        sredni_tauCFF_contra.iloc[n-1] = np.nanmean(dataset.TCFF_CONTRA[old])
        sredni_tauPFF_ipsi.iloc[n-1] = np.nanmean(dataset.TPFF_IPSI[old])
        sredni_tauPFF_contra.iloc[n-1] = np.nanmean(dataset.TPFF_CONTRA[old])
        sredni_FV_ipsi.iloc[n-1] = np.nanmean(dataset.FV_IPSI[old])
        sredni_FV_contra.iloc[n-1] = np.nanmean(dataset.FV_CONTRA[old])
        CV.iloc[n-1]=min(dataset.CV[old])
     else:
        sredni_tauCFF_ipsi.iloc[n-1] = np.nan
        sredni_tauCFF_contra.iloc[n-1] = np.nan
        sredni_tauPFF_ipsi.iloc[n-1] = np.nan
        sredni_tauPFF_contra.iloc[n-1] = np.nan
        sredni_FV_ipsi.iloc[n-1] = np.nan
        sredni_FV_contra.iloc[n-1] = np.nan
        
        before = (dataset.NR == n) & (dataset.DURING == 0)
        
        before_tauCFF_ipsi.iloc[n-1] = np.nanmean(dataset.TCFF_IPSI[before])
        before_tauCFF_contra.iloc[n-1] = np.nanmean(dataset.TCFF_CONTRA[before])
        before_tauPFF_ipsi.iloc[n-1] = np.nanmean(dataset.TPFF_IPSI[before])
        before_tauPFF_contra.iloc[n-1] = np.nanmean(dataset.TPFF_CONTRA[before])
        before_FV_ipsi.iloc[n-1] = np.nanmean(dataset.FV_IPSI[before])
        before_FV_contra.iloc[n-1] = np.nanmean(dataset.FV_CONTRA[before])
        
        during = (dataset.NR == n) & (dataset.DURING == 1)
        
        during_tauCFF_ipsi.iloc[n-1] = np.nanmean(dataset.TCFF_IPSI[during])
        during_tauCFF_contra.iloc[n-1] = np.nanmean(dataset.TCFF_CONTRA[during])
        during_tauPFF_ipsi.iloc[n-1] = np.nanmean(dataset.TPFF_IPSI[during])
        during_tauPFF_contra.iloc[n-1] = np.nanmean(dataset.TPFF_CONTRA[during])
        during_FV_ipsi.iloc[n-1] = np.nanmean(dataset.FV_IPSI[during])
        during_FV_contra.iloc[n-1] = np.nanmean(dataset.FV_CONTRA[during])
        
        CV.iloc[n-1]=max(dataset.CV[old])
        
        
        
     n = n+1    
    
results_final= pd.DataFrame(index=np.arange(97), columns=np.arange(18))

frames = [sredni_tauCFF_ipsi,sredni_tauCFF_contra,sredni_tauPFF_ipsi,
sredni_tauPFF_contra,sredni_FV_ipsi,sredni_FV_contra,before_tauCFF_ipsi, before_tauCFF_contra,
before_tauPFF_ipsi,before_tauPFF_contra, before_FV_ipsi,before_FV_contra,       
during_tauCFF_ipsi,during_tauCFF_contra,during_tauPFF_ipsi,during_tauPFF_contra,
during_FV_ipsi, during_FV_contra, CV]

results_final=pd.concat(frames, axis=1)
results_final.columns=(['sredni_tauCFF_ipsi','sredni_tauCFF_contra','sredni_tauPFF_ipsi',
'sredni_tauPFF_contra','sredni_FV_ipsi','sredni_FV_contra','before_tauCFF_ipsi', 'before_tauCFF_contra',
'before_tauPFF_ipsi','before_tauPFF_contra', 'before_FV_ipsi','before_FV_contra',       
'during_tauCFF_ipsi','during_tauCFF_contra','during_tauPFF_ipsi','during_tauPFF_contra',
'during_FV_ipsi', 'during_FV_contra', 'CV'])
    
results_final.to_csv(SOURCE_FILE_PATH + '/' +'results_before_during.csv', sep = ';', index = False)
    