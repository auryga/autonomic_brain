# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 13:47:49 2021

@author: agnie
"""
from pylab import rcParams

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
        
        
 
   
   
         
    return dataset



name="C:\Moje_dokumenty\Po_doktoracie_06_2019_08_2021\zaproszony_artykul\PAC11.csv"



dataset= read_file(name, separator = ';', decimal_sign = ',', delete_column = 'DateTime')
rso2_left = dataset.iloc[:,7]
rso2_right = dataset.iloc[:,8]
abp= dataset.iloc[:,2]
brs = dataset.iloc[:,3]
TOxL = dataset.iloc[:,0]
co=dataset.iloc[:,6]
ci=dataset.iloc[:,5]

rso2_left=rso2_left.apply(lambda x: np.where(x < 50,np.nan,x))
rso2_right=rso2_right.apply(lambda x: np.where(x < 50,np.nan,x))
brs=brs.apply(lambda x: np.where(x > 35,np.nan,x))

fig = plt.figure()
rcParams['figure.figsize'] = 40, 20
plt.rcParams['font.size'] = '16'

ax=plt.subplot(1, 1, 1)
a=5700
b=5730
plt.axvspan(a, b, color='y', alpha=0.5, lw=0)
plt.axvspan(a+5, b+5, color='y', alpha=0.5, lw=0)
plt.plot(rso2_left)
plt.xlim([5700,6000])
plt.ylim([55,80])
plt.axline((5700, 65), (6000, 65),color="red")
x_ticks = np.arange(5700, 6030,30)
plt.xticks(x_ticks)
#ax.set_title("rso2 ipsilateral")
#ax.set_xlabel("Time [s]",fontsize=14)
ax.set_ylabel("rSO$_{2}$ ipsilateral [%]",fontsize=16)
ax.grid(True)
plt.title("Moving-average window to get regional decreased rSO$_{2}$")


