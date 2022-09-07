# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:46:15 2021

@author: agnie
"""
# pooled data - matrix of dummy variables

import pandas as pd
import numpy as np



puste=pd.DataFrame(index=np.arange(140), columns=np.arange(35))

start=0
for i in range (0,35):
    puste.iloc[start:start+4,i]=1
    i+=1
    start+=4
    
puste.fillna(0)