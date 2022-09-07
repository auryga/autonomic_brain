# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:40:20 2021

@author: agnie

"""










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
        results_final['Ca_PFF_ipsi'].values[n] = results_final['Ca2_PFF_L'].values[n]
        results_final['Ca_PFF_contra'].values[n] = results_final['Ca2_PFF_R'].values[n]
        results_final['T_CFF_ipsi'].values[n] = results_final['T_CFF2L'].values[n]
        results_final['T_CFF_contra'].values[n] = results_final['T_CFF2R'].values[n]
        results_final['T_PFF_ipsi'].values[n] = results_final['T_PFF2L'].values[n]
        results_final['T_PFF_contra'].values[n] = results_final['T_PFF2R'].values[n]
        results_final['FV_ipsi'].values[n] = results_final['FVL'].values[n]
        results_final['FV_contra'].values[n] = results_final['FVR'].values[n]
        results_final['AmpFV_ipsi'].values[n] = results_final['AmpFVL'].values[n]
        results_final['AmpFV_contra'].values[n] = results_final['AmpFVR'].values[n]
             
        
    elif side_patient == 2:
        results_final['CVR_ipsi'].values[n] = results_final['CVR2R'].values[n]
        results_final['CVR_contra'].values[n] = results_final['CVR2L'].values[n]
        results_final['Ca_CFF_ipsi'].values[n] = results_final['Ca_CFF1R'].values[n]
        results_final['Ca_CFF_contra'].values[n] = results_final['Ca_CFF1L'].values[n]
        results_final['Ca_PFF_ipsi'].values[n] = results_final['Ca2_PFF_R'].values[n]
        results_final['Ca_PFF_contra'].values[n] = results_final['Ca2_PFF_L'].values[n]
        results_final['T_CFF_ipsi'].values[n] = results_final['T_CFF2R'].values[n]
        results_final['T_CFF_contra'].values[n] = results_final['T_CFF2L'].values[n]
        results_final['T_PFF_ipsi'].values[n] = results_final['T_PFF2R'].values[n]
        results_final['T_PFF_contra'].values[n] = results_final['T_PFF2L'].values[n]
        results_final['FV_ipsi'].values[n] = results_final['FVR'].values[n]
        results_final['FV_contra'].values[n] = results_final['FVL'].values[n]
        results_final['AmpFV_ipsi'].values[n] = results_final['AmpFVR'].values[n]
        results_final['AmpFV_contra'].values[n] = results_final['AmpFVL'].values[n]
        
    elif side_patient == 3:
        CVR = np.array([results_final['CVR2R'].values[n],results_final['CVR2L'].values[n]])
        if np.isnan(CVR).any() == True:
            results_final['CVR_ipsi'].values[n] = np.nansum(CVR)
        else:
            results_final['CVR_ipsi'].values[n] = np.nansum(CVR)/2
        results_final['CVR_contra'].values[n] = np.nan
        
        CA = np.array([results_final['Ca_CFF1R'].values[n], results_final['Ca_CFF1L'].values[n]])
        if np.isnan(CA).any() == True:
            results_final['Ca_CFF_ipsi'].values[n] = np.nansum(CA)
        else:
             results_final['Ca_CFF_ipsi'].values[n] = np.nansum(CA)/2
        results_final['Ca_CFF_contra'].values[n] = np.nan
        
       
        CAP = np.array([results_final['Ca2_PFF_R'].values[n],results_final['Ca2_PFF_L'].values[n]])
        if np.isnan(CAP).any() == True:
            results_final['Ca_PFF_ipsi'].values[n] = np.nansum(CAP)
        else:
            results_final['Ca_PFF_ipsi'].values[n] = np.nansum(CAP)/2
        results_final['Ca_PFF_contra'].values[n] = np.nan
        
        TCF =  np.array([results_final['T_CFF2R'].values[n], results_final['T_CFF2L'].values[n]])
        if np.isnan(TCF).any() == True:
            results_final['T_CFF_ipsi'].values[n] = np.nansum(TCF)
        else:
            results_final['T_CFF_ipsi'].values[n] = np.nansum(TCF)/2
        results_final['T_CFF_contra'].values[n] = np.nan
        
        
        TPF = np.array([results_final['T_PFF2R'].values[n], results_final['T_PFF2L'].values[n]])
        if np.isnan(TPF).any() == True:
            results_final['T_PFF_ipsi'].values[n] = np.nansum(TPF)
        else:
            results_final['T_PFF_ipsi'].values[n] = np.nansum(TPF)/2
        results_final['T_PFF_contra'].values[n] = np.nan
        
        FVI = np.array([results_final['FVR'].values[n] , results_final['FVL'].values[n]])
        if np.isnan(FVI).any() == True:
            results_final['FV_ipsi'].values[n] = np.nansum(FVI)
        else:
            results_final['FV_ipsi'].values[n] = np.nansum(FVI)/2
        results_final['FV_contra'].values[n] = np.nan
        
        FVC = np.array([results_final['AmpFVR'].values[n] , results_final['AmpFVL'].values[n]])
        if np.isnan(FVC).any() == True:
            results_final['AmpFV_ipsi'].values[n] = np.nansum(FVC)
        else:
            results_final['AmpFV_ipsi'].values[n] = np.nansum(FVC)/2
        results_final['AmpFV_contra'].values[n] = np.nan
        
        
        
        
    else:
        results_final['CVR_ipsi'].values[n] = np.nan
        results_final['CVR_contra'].values[n] = np.nan
        results_final['Ca_CFF_ipsi'].values[n] = np.nan
        results_final['Ca_CFF_contra'].values[n] = np.nan
        results_final['Ca_PFF_ipsi'].values[n] = np.nan
        results_final['Ca_PFF_contra'].values[n] = np.nan
        results_final['T_CFF_ipsi'].values[n] = np.nan
        results_final['T_CFF_contra'].values[n] = np.nan
        results_final['T_PFF_ipsi'].values[n] = np.nan
        results_final['T_PFF_contra'].values[n] = np.nan
        results_final['FV_ipsi'].values[n] = np.nan
        results_final['FV_contra'].values[n] = np.nan
        results_final['AmpFV_ipsi'].values[n] = np.nan
        results_final['AmpFV_contra'].values[n] = np.nan

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