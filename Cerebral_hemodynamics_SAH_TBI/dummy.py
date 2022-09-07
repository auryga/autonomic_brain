# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:46:15 2021

@author: agnie
"""
# pooled data - matrix of dummy variables

SOURCE_FILE_PATH="C:\Moje_dokumenty\Po_doktoracie\EMBC"

import pandas as pd
import numpy as np



puste=pd.DataFrame(0,index=np.arange(80), columns=np.arange(20))


start=0
for i in range (0,20):
    puste.iloc[start:start+4,i]=1
    i+=1
    start+=4
    


puste.to_csv(SOURCE_FILE_PATH + '/' +'dummy.csv', sep = ';', index = False)
    