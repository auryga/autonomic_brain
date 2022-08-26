# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:45:07 2022

@author: agnie
"""

    
       
    
    
    #######drop out values of BRS-ICP where are np.nan, i.e. in both
    ######BRS and ICP needs to be data
    binary_mask_brs = dataset['brs']>0
    binary_mask_icp = dataset['ICP']>0
    
    binary_mask_all = binary_mask_brs*binary_mask_icp
    
    dataset_all=dataset[binary_mask_all]
    
    
   
    
    
    
    
    
    
    
    
    
  
    
    #############  reindeksowanie macierzy po synchronizacji- tam gdzie sygnał w obu
    
    dataset_all=dataset_all.reset_index()
    
    ############interpolacja nanów
    filled_i = interpolate_gaps(dataset_all.iloc[:,1], limit=6)
    filled_b = interpolate_gaps(dataset_all.iloc[:,2], limit=6)
    dataset_i = pd.DataFrame(filled_i)
    dataset_b = pd.DataFrame(filled_b)
    frames = [dataset_i, dataset_b]
    result_all_m = pd.concat(frames,axis=1)
    result_all_m.columns = ['ICP', 'BRS']
    result_all_m = result_all_m.dropna() #####sklejamy
    ############  reindeksowanie macierzy po synchronizacji i interpolacji
    result_all_m = result_all_m.reset_index()
    result_all_m .drop(['index'], axis=1, inplace=True)
    
    
    
    ################graph to show interpolate data without nan's
    f,(ax1,ax2)=plt.subplots(2,1,figsize=(10,5),sharex='col')
    ax1.plot(result_all_m['BRS'])
    ax2.plot(result_all_m['ICP'])
    ax1.set(ylabel='ICP [mm Hg]')
    ax2.set(ylabel = 'BRS [ms/mm Hg]')
    ax1.set(title=f"BRS and ICP without outliers and nan's and interpolate")
    ax2.set(xlabel = 'Time')
    points = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500]
    points2 = []
    for i in range(0,len(points)):
        points2.append(points[i]/360)
    points2 = [round(elem, 2) for elem in points2 ]
    ax2.set_xticks(points)
    ax2.set_xticklabels(points2)
    
    
        
    ############### mediana ruchoma co 5 minut
    overall_pearson_r = result_all_m['BRS'].corr(result_all_m['ICP'])
    r, p = stats.pearsonr(result_all_m['BRS'], result_all_m['ICP'])
    r1, p1 = stats.spearmanr(result_all_m['BRS'], result_all_m['ICP'])
       
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    print(f"Scipy computed Spearman r: {r1} and p-value: {p1}")
    
    f,ax=plt.subplots(figsize=(10,5))
    result_all_m.rolling(window=180,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time',ylabel='mm Hg or ms/mm Hg')
    ax.set(title=f"Scipy computed Pearson r: {r:.2f} and p-value: {p:.2f}\n \
           Scipy computed Spearman r: {r1:.2f} and p-value: {p1:.2f}")
    points = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500]
    points2 = []
    for i in range(0,len(points)):
        points2.append(points[i]/360)
    points2 = [round(elem, 2) for elem in points2 ]
    ax.set_xticks(points)
    ax.set_xticklabels(points2)
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    