# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:48:40 2020

@author: liyifu
"""


import os

os.chdir(r"G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal")

from MTLUpdate import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
from sklearn.metrics import mean_squared_error
from math import log
from sklearn.model_selection import KFold
import pandas as pd
from Network_IBOSS_and_Benchmarks import *
#from pyspc import *

allBetaEstimationError = list()
    

totalNumIte = 100
meanTimeAgg = np.empty([0,11])
sampleSizeUsedAvgAgg = np.empty([0,54])
RMSESummary = np.empty([0,10])
RMSEPredSummary = np.empty([0,10])

for n in [5000,50000]:
    
    for similarity in [5,10]:
        
        for modelSparsity in [0.3,0.7]:
        
            for signal_to_noise_ratio in [3,7]:
                
                [sampleSizeUsedAvg, 
                 MeanRMSE, SERMSE,
                 MeanPredRMSE, SEPredRMSE,TimeAllSum]=Network_IBOSS_and_Benchmarks(n, similarity, 
                                 modelSparsity, signal_to_noise_ratio, totalNumIte)

                RMSESummary = np.vstack((
                                         RMSESummary,
                                         np.hstack(([n,np.mean(sampleSizeUsedAvg[1,:]),similarity,modelSparsity,signal_to_noise_ratio],MeanRMSE)),
                                         np.hstack(([n,np.mean(sampleSizeUsedAvg[1,:]),similarity,modelSparsity,signal_to_noise_ratio],SERMSE))
                                         ))
                                         
                RMSEPredSummary = np.vstack((
                                        RMSEPredSummary,
                                        np.hstack(([n,np.mean(sampleSizeUsedAvg[1,:]),similarity,modelSparsity,signal_to_noise_ratio],MeanPredRMSE)),
                                        np.hstack(([n,np.mean(sampleSizeUsedAvg[1,:]),similarity,modelSparsity,signal_to_noise_ratio],SEPredRMSE))
                                        ))
                
                sampleSizeUsedAvgAgg = np.vstack((
                                        sampleSizeUsedAvgAgg,
                                        np.hstack(([n,similarity,modelSparsity,signal_to_noise_ratio],sampleSizeUsedAvg.reshape(-1)))
                    
                                        ))
                
                meanTimeAgg = np.vstack((meanTimeAgg,
                                         np.mean(np.array(TimeAllSum), axis=0)))
                with pd.ExcelWriter(r'G:\Other computers\CEC Desktop\Desktop\IISE Transaction\1st rebuttal\SimulationResultSummary_6_2.xlsx') as writer:
                                        
                    pd.DataFrame(RMSESummary).to_excel(writer, 
                        sheet_name='RMSE')
                
                    pd.DataFrame(RMSEPredSummary).to_excel(writer, 
                        sheet_name='RMSPE')
                
                    pd.DataFrame(sampleSizeUsedAvgAgg).to_excel(writer, 
                        sheet_name='Sample Size Summary')
                    
                    pd.DataFrame(meanTimeAgg).to_excel(writer, 
                        sheet_name='Time Taken')
#print(np.vstack((np.array(MSEAll).reshape(1,-1),np.array(sampleSizeUsed))))
#
#
#### IBOSS one system at a time, no sample limit
#betaFit = np.zeros([p+1,numSystems])
#betaFitPre = np.zeros([p+1,numSystems])
#sAllONIBOSS = list()
#maxN = np.max(sAll)
#for k in np.arange(0,numSystems):
#    diffBetaAgg = list()
#    
#    
#    for s in np.arange(1,maxN):
#    #    totalDataExtracted = list()
#        ## IBOSS Algorithm with beta change
#        totalDataToBeReduce = totalData[:,:,k]
#        
#        xySampled = IBOSS(s,totalDataToBeReduce)
#    #    totalDataExtracted.append(xySampled)
#        xSampled = xySampled[:,:-1]
#        ySampled = xySampled[:,-1]
#           
#        betaFitPre[:,k] = betaFit[:,k]
#        betaFit[:,k] = np.dot(np.dot(np.linalg.inv(np.dot(xSampled.T,xSampled)),xSampled.T),ySampled)
#        
#        diffBeta = mean_squared_error(betaFitPre[:,k],betaFit[:,k])
#        if s >1:
#            diffBetaAgg.append(diffBeta/mean_squared_error(betaFitPre[:,k],np.zeros(p+1)))
#            if diffBetaAgg[-1]<1e-3:
#                break
#                   
#    sAllONIBOSS.append(s)   
#
#methodName.append('IBOSS One System At a Time No Sample limit') 
#MSEAll.append(mean_squared_error(betaFit, trueBeta))
#sampleSizeUsed.append(np.array(sAllONIBOSS)*2*p)
#
#
#### Random Sampling one system at a time, no sample limit
#betaFit = np.zeros([p+1,numSystems])
#betaFitPre = np.zeros([p+1,numSystems])
#sAllONRnd = list()
#MaxN = np.max(sAll)*2*p
#for k in np.arange(0,numSystems):
#    diffBetaAgg = list()
#    
#    
#    for sampleSize in np.arange(1,MaxN):
#    #    totalDataExtracted = list()
#        ## IBOSS Algorithm with beta change
#        totalDataToBeReduce = totalData[:,:,k]
#        
#        xySampled = randomSampling(sampleSize,totalDataToBeReduce)
#    
#    #    totalDataExtracted.append(xySampled)
#        xSampled = xySampled[:,:-1]
#        ySampled = xySampled[:,-1]
#           
#        betaFitPre[:,k] = betaFit[:,k]
#        betaFit[:,k] = np.dot(np.dot(np.linalg.inv(np.dot(xSampled.T,xSampled)),xSampled.T),ySampled)
#        
#        diffBeta = mean_squared_error(betaFitPre[:,k],betaFit[:,k])
#        if sampleSize >1:
#            diffBetaAgg.append(diffBeta/mean_squared_error(betaFitPre[:,k],np.zeros(p+1)))
#            if diffBetaAgg[-1]<1e-3:
#                break
#                   
#    sAllONRnd.append(sampleSize)   
#
#methodName.append('Random Sampling One System At a Time No Sample limit') 
#MSEAll.append(mean_squared_error(betaFit, trueBeta))
#sampleSizeUsed.append(np.array(sAllONRnd))

