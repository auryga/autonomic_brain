
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 22:51:51 2021

@author: agnie


"""
from pylab import rcParams

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300




def read_file(file_name, separator = None, decimal_sign = None, delete_column = None):
    dataset = pd.read_csv (file_name, sep = separator, decimal= decimal_sign)
    dataset = pd.DataFrame(dataset)
    if delete_column != None:
        del dataset[delete_column]
        
        
 
   
   
         
    return dataset



name="C:\Moje_dokumenty\Po_doktoracie_09_2021\Miniatura\wyniki_17_1\PAC_02 - SAH_20141103114045_ICM_HP_r2.csv"



dataset= read_file(name, separator = ';', decimal_sign = ',', delete_column = 'DateTime')

brs = dataset.iloc[:,2]
icp = dataset.iloc[:,1]



brs=brs.apply(lambda x: np.where(x<1,np.nan,x))



a=1530
b=1830
fig = plt.figure()
rcParams['figure.figsize'] = 40, 20
#plt.rcParams['font.size'] = '16'

ax=plt.subplot(2, 1, 1)
plt.axvspan(1530,1610 , color='red', alpha=0.3, lw=0)
plt.axvspan(1610,1830, color='g', alpha=0.3, lw=0)


plt.plot(rso2_left,linewidth=4)
plt.xlim([a,b])
plt.ylim([58,72])
plt.axline((a, 65), (b, 65),color="red")
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
plt.rcParams.update({'font.size': 22})
ax.set_xticklabels(['0','30','1:00','1:30','2:00','2:30','3:00','4:00','4:30','5:00'])


#ax.set_title("rso2 ipsilateral")
#ax.set_xlabel("Time [s]",fontsize=14)

ax.set_ylabel("rSO$_{2}$ [%]",fontsize=28)

ax.grid(True)
plt.title("Moving-average window to obtain reduced rSO$_{2}$", fontsize =35)
plt.text(1568, 70, 'during rCD', fontsize=35, color='black')
plt.text(1740, 70, 'baseline', fontsize=35, color='black')


ax=plt.subplot(5, 1, 2)

plt.axvspan(1530,1610 , color='red', alpha=0.3, lw=0)
plt.axvspan(1610,1830, color='g', alpha=0.3, lw=0)

plt.axvspan(a,a+30 , color='red', alpha=0.0, lw=1)
plt.axvspan(a+5,a+30+5 , color='red', alpha=0.0, lw=1)
#plt.axvline(x=1530, color='black')
#plt.axvline(x=1535, color='black')
#plt.axvline(x=1560, color='black')
#plt.axvline(x=1565, color='black')

plt.plot(co,linewidth=4)
plt.xlim([a,b])
plt.ylim([2.5,4.5])
#ax.set_xlabel("Time [s]",fontsize=14)
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_ylabel("CO [L/min]",fontsize=28)
ax.grid(True)
ax.set_xticklabels(['0','30','1:00','1:30','2:00','2:30','3:00','4:00','4:30','5:00'])


ax=plt.subplot(5, 1, 3)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.axvspan(1530,1610 , color='red', alpha=0.3, lw=0)
plt.axvspan(1610,1830, color='g', alpha=0.3, lw=0)

plt.axvspan(a,a+30 , color='red', alpha=0.0, lw=1)
plt.axvspan(a+5,a+30+5 , color='red', alpha=0.0, lw=1)
#plt.axvline(x=1530, color='black')
#plt.axvline(x=1535, color='black')
#plt.axvline(x=1560, color='black')
#plt.axvline(x=1565, color='black')

plt.plot(ci, linewidth=4)
plt.xlim([a,b])
plt.ylim([1.5,2.5])
#ax.set_xlabel("Time [s]",fontsize=14)
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_ylabel("CI [L/min/m$^{2}$]",fontsize=28)
ax.grid(True)
ax.set_xticklabels(['0','30','1:00','1:30','2:00','2:30','3:00','4:00','4:30','5:00'])


ax=plt.subplot(5, 1, 4)






plt.axvspan(1530,1610 , color='red', alpha=0.3, lw=0)
plt.axvspan(1610,1830, color='g', alpha=0.3, lw=0)

plt.axvspan(a,a+30 , color='red', alpha=0.0, lw=1)
plt.axvspan(a+5,a+30+5 , color='red', alpha=0.0, lw=1)
#plt.axvline(x=1530, color='black')
#plt.axvline(x=1535, color='black')
#plt.axvline(x=1560, color='black')
#plt.axvline(x=1565, color='black')

plt.plot(TOxL, linewidth=4)
plt.xlim([a,b])
plt.ylim([-1,1])
#ax.set_xlabel("Time [s]",fontsize=14)
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_ylabel("TOxa [a.u.]",fontsize=28)
ax.grid(True)
ax.set_xticklabels(['0','30','1:00','1:30','2:00','2:30','3:00','4:00','4:30','5:00'])


ax=plt.subplot(5, 1, 5)

plt.axvspan(1530,1610 , color='red', alpha=0.3, lw=0)
plt.axvspan(1610,1830, color='g', alpha=0.3, lw=0)

plt.axvspan(a,a+30 , color='red', alpha=0.0, lw=1)
plt.axvspan(a+5,a+30+5 , color='red', alpha=0.0, lw=1)
#plt.axvline(x=1530, color='black')
#plt.axvline(x=1535, color='black')
#plt.axvline(x=1560, color='black')
#plt.axvline(x=1565, color='black')


plt.plot(brs, linewidth=4)
plt.xlim([a,b])
plt.ylim([1,35])
x_ticks = np.arange(a,b,30)
plt.xticks(x_ticks)
ax.set_xlabel("Time [s]",fontsize=35)
ax.set_ylabel("BRS [ms/mm Hg]", fontsize=28)
ax.grid(True)
ax.set_xticklabels(['0','30','1:00','1:30','2:00','2:30','3:00','4:00','4:30','5:00'])



plt.show()



fig.savefig('temp.png', dpi=fig.dpi)