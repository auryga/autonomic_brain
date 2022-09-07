# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 22:51:51 2021

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



name="C:\Moje_dokumenty\Po_doktoracie\zaproszony_artykul\PAC34.csv"



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
brs=brs.apply(lambda x: np.where(x<1,np.nan,x))



a=1530
b=1830
fig = plt.figure()
rcParams['figure.figsize'] = 40, 20
plt.rcParams['font.size'] = '16'

ax=plt.subplot(5, 1, 1)
plt.axvspan(a, a+30, color='y', alpha=0.5, lw=0)
plt.axvspan(a+5, a+35, color='y', alpha=0.5, lw=0)
plt.plot(rso2_left)
plt.xlim([a,b])
plt.ylim([58,72])
plt.axline((a, 65), (b, 65),color="red")
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
#ax.set_title("rso2 ipsilateral")
#ax.set_xlabel("Time [s]",fontsize=14)
ax.set_ylabel("rSO$_{2}$ ipsilateral [%]",fontsize=16)
ax.grid(True)
plt.title("Moving-average window to get regional decreased rSO$_{2}$")



ax=plt.subplot(5, 1, 2)
plt.axvspan(a, a+30, color='y', alpha=0.5, lw=0)
plt.axvspan(a+5, a+35, color='y', alpha=0.5, lw=0)
plt.plot(co)
plt.xlim([a,b])
plt.ylim([2.5,4.5])
#ax.set_xlabel("Time [s]",fontsize=14)
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_ylabel("CO [L/min]",fontsize=16)
ax.grid(True)


ax=plt.subplot(5, 1, 3)
plt.axvspan(a, a+30, color='y', alpha=0.5, lw=0)
plt.axvspan(a+5, a+35, color='y', alpha=0.5, lw=0)
plt.plot(ci)
plt.xlim([a,b])
plt.ylim([1.5,2.5])
#ax.set_xlabel("Time [s]",fontsize=14)
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_ylabel("CI [L/min/m$^{2}$]",fontsize=16)
ax.grid(True)


ax=plt.subplot(5, 1, 4)
plt.axvspan(a, a+30, color='y', alpha=0.5, lw=0)
plt.axvspan(a+5, a+35, color='y', alpha=0.5, lw=0)
plt.plot(TOxL)
plt.xlim([a,b])
plt.ylim([-1,1])
#ax.set_xlabel("Time [s]",fontsize=14)
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_ylabel("TOxa ipsilateral [a.u.]",fontsize=16)
ax.grid(True)


ax=plt.subplot(5, 1, 5)
plt.axvspan(a, a+30, color='y', alpha=0.5, lw=0)
plt.axvspan(a+5, a+35, color='y', alpha=0.5, lw=0)
plt.plot(brs)
plt.xlim([a,b])
plt.ylim([1,35])
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_xlabel("Time [s]",fontsize=16)
ax.set_ylabel("BRS [mm Hg]", fontsize=16)
ax.grid(True)



plt.show()