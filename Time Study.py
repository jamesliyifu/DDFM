# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
os.chdir(r"G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal")
from sklearn import preprocessing
from Network_IBOSS_and_Benchmarks import *
import warnings
warnings.filterwarnings("ignore")

furnance_index = [223,268,279,282,288]

outfile = 'G:\\Other computers\\CEC Desktop\\Desktop\\IISE Transaction\\1st rebuttal\\danjing\\'


totalNumIte=100

    
with pd.ExcelWriter(r'G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal\RealDataResultSummaryDist_6_3.xlsx') as writer:
     
    [sampleSizeUsedAvg, MeanRMSE, SERMSE,TimeAllAgg, BetaAllAgg] = Network_RealData_DistN(outfile, furnance_index, totalNumIte)
    
    RMSESummary = np.vstack((MeanRMSE,SERMSE))
                    
    pd.DataFrame(RMSESummary).to_excel(writer, 
    sheet_name='RMSE Real Data Result')
    

    pd.DataFrame(sampleSizeUsedAvg).to_excel(writer, 
        sheet_name='Size Real Data Result')
    
                        
    meanTimeAgg = np.mean(np.array(TimeAllAgg), axis=0)
    
    pd.DataFrame(meanTimeAgg).to_excel(writer, 
        sheet_name='Time Taken')    
    
    np.save(r'G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal\RealDataBeta_6_2.npy', BetaAllAgg)
    
    
# Drawing    
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
RealBeta = np.load(r'G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal\RealDataBeta_6_2.npy',allow_pickle=True)
# RealBeta = np.mean(np.array(RealBeta), axis=0)

for i in range(5):
    
    for j in range(5):
        filtered = list()
        for k in range(19):
            
            arr1 = RealBeta[:,j,i,k,0]
            q1 = np.quantile(arr1, 0.25)
             
            # finding the 3rd quartile
            q3 = np.quantile(arr1, 0.75)
            med = np.median(arr1)
             
            # finding the iqr region
            iqr = q3-q1
             
            # finding upper and lower whiskers
            upper_bound = q3+(1.5*iqr)
            lower_bound = q1-(1.5*iqr)
            outliers = arr1[(arr1 <= lower_bound) | (arr1 >= upper_bound)]
            arr2 = arr1[(arr1 >= lower_bound) & (arr1 <= upper_bound)]
            filtered.append(arr2)
            
        filtered = np.array(filtered).transpose()
    
        fig = plt.figure(figsize =(10, 7))
         
        # Creating axes instance
        ax = fig.add_axes([0, 0, 1, 1])
         
        # Creating plot
        bp = ax.boxplot(filtered)
         
        # show plot
        plt.show()
        plt.tight_layout()    
        
        
################
from sklearn import preprocessing
with pd.ExcelWriter(r'G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal\RealDataVariableSummary_6_3.xlsx') as writer:
     
    for i in range(5):
        
        filteredAll = list()
        for j in range(5):
            
            filtered = list()
            for k in range(19):
                
                arr1 = RealBeta[:,j,k,i]
                filtered.append(np.hstack((np.mean(arr1),np.array(st.norm.interval(alpha=0.95, loc=np.mean(arr1), scale=st.sem(arr1))))))
                
            filteredAll.append(filtered)     
                
        filteredAll = np.nan_to_num(np.array(filteredAll))

        
        
        forPrint = np.empty([19,0])
        for j in range(5):
            forPrint = np.hstack((forPrint,preprocessing.normalize(filteredAll[j,:,:])))
            
        pd.DataFrame(forPrint).to_excel(writer, 
            sheet_name='System '+str(i+1))
        
        # filtered = np.array(filtered).transpose()
    
        # fig = plt.figure(figsize =(10, 7))
         
        # # Creating axes instance
        # ax = fig.add_axes([0, 0, 1, 1])
         
        # # Creating plot
        # bp = ax.boxplot(filtered)
         
        # # show plot
        # plt.show()
        # plt.tight_layout()    
        

























        