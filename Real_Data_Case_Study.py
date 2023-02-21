

import numpy as np
from pathlib import Path
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
os.chdir(r"C:\Users\liyifu\Desktop\Distrbuted Filtering and Modeling")
import warnings
from sklearn import preprocessing
from Network_IBOSS_and_Benchmarks import *
os.environ["PYTHONWARNINGS"] = "ignore"

plotLocation = 'C://Users//liyifu//Desktop//Distrbuted Filtering and Modeling//plot//'


# furnance_index = [223,268,271,279,282,288]
furnance_index = [223,268,279,282,288]

outfile = 'C:\\Users\\liyifu\Desktop\\Distrbuted Filtering and Modeling\\danjing\\'

trteP = 0.9
totalVar = 56
nAlpha = 50
indDel = [1,3,4,5,6,7,8,9,11,12,13,15,16,17,18,19,21,23,24,27,28,29,30,31,34,37,38,39,40,42,43,45,47,49,50,51,52]
  
# num_trancated = 500
# numSampleUsedPerSystem = 1
totalTrainData = list()
totalTestData = list()

for i in range(len(furnance_index)):
    currData = np.load(outfile+'\\'+str(furnance_index[i])+'_data.npy',allow_pickle=True)

    xTrainAll = np.empty([0,totalVar-len(indDel)])
    xTestAll = np.empty([0,totalVar-len(indDel)])
    yTrainAll = np.empty(0)
    yTestAll = np.empty(0)    
    for j in range(len(currData)):

    # for j in range(1):    
        dataImported = currData[j][:,:totalVar]
        if dataImported.shape[0]<1500:
            continue
   
            
        y = np.array(dataImported[:,34],dtype=float)[1:]
        x = np.array(np.delete(dataImported, indDel, axis=1),dtype=float)[1:,:]
     
        n = x.shape[0]

        
        
        shuffleInd = np.arange(n)
        
        np.random.shuffle(shuffleInd)
        
        xTrain, xTest = x[shuffleInd[:int(n*trteP)],:], x[shuffleInd[int(n*trteP):],:]
        yTrain, yTest = y[shuffleInd[:int(n*trteP)]], y[shuffleInd[int(n*trteP):]]
        
        xTrainAll = np.vstack((xTrainAll,xTrain))
        # scaler = preprocessing.StandardScaler().fit(xTrainAll)
        xTestAll = np.vstack((xTestAll,xTest))
        
        yTrainAll = np.hstack((yTrainAll,yTrain))
        yTestAll = np.hstack((yTestAll,yTest))
        
    yTestAll = yTestAll-np.mean(yTrainAll)
    yTrainAll = yTrainAll-np.mean(yTrainAll)
    
    totalTrainData.append(np.hstack((xTrainAll,
                                     yTrainAll.reshape(-1,1))))
    
    totalTestData.append(np.hstack((xTestAll,
                                    yTestAll.reshape(-1,1))))
                    
            

# IBOSS Network
p = totalTrainData[0].shape[1]-1
numSystems = len(totalTrainData)

systemName = list()
betaName = list()
for i in range(numSystems):
    systemName.append('Sys '+str(i+1))
for i in range(p):
    betaName.append('beta '+str(i+1))
        
sampleSizeUsed = list()

betaFit = np.zeros([p,numSystems])
betaFitPre = np.zeros([p,numSystems])

diffBetaAgg = list()

sAll = list()
methodName = list()
yTestAll = np.empty(0)
yHatAll = np.empty(0)


k=0
n = totalTrainData[k].shape[0]
maxN = int(n/2/p)
RMSEAll = list()
for s in np.arange(1,maxN):
#    totalDataExtracted = list()
    ## IBOSS Algorithm with beta change
    totalDataToBeReduce = totalTrainData[k]
    
    
    xySampled = IBOSS(s,totalDataToBeReduce)
    
    xSampled = xySampled[:,:-1]
    ySampled = xySampled[:,-1]
       
    betaFitPre[:,k] = betaFit[:,k]
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
        fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                     
    betaFit[:,k] = reg.coef_
    # betaFit[:,k] = FiveFoldCV(xSampled,ySampled)
    
    diffBeta = mean_squared_error(betaFitPre[:,k],betaFit[:,k])
    if s >1:
        diffBetaAgg.append(diffBeta/mean_squared_error(betaFitPre[:,k],np.zeros(p)))
        
        if diffBetaAgg[-1]<1e-3:
            break
        
        
yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
    

    
sAll.append(s)       

        
for k in np.arange(1,numSystems):
    diffBetaAgg = list()
    n = totalTrainData[k].shape[0]
    maxN = int(n/2/p)

    for s in np.arange(1,maxN):
    #    totalDataExtracted = list()
        ## IBOSS Algorithm with beta change
        totalDataToBeReduce = totalTrainData[k]
        
    
        xySampled = IBOSS(s,totalDataToBeReduce)
        
    #    totalDataExtracted.append(xySampled)
        xSampled = xySampled[:,:-1]
        ySampled = xySampled[:,-1]
        yResi = ySampled-np.dot(xSampled,betaFit[:,0])
        
        # betaAdd = FiveFoldCV(xSampled,yResi)
        
        reg = LassoOneSE(xSampled,yResi)
        betaAdd = reg.coef_

        betaFitPre[:,k] = betaFit[:,k]
        betaFit[:,k] = betaFit[:,0]+betaAdd
        

        diffBeta = mean_squared_error(betaFitPre[:,k],betaFit[:,k])
        if s >1:
            diffBetaAgg.append(diffBeta/mean_squared_error(betaFitPre[:,k],np.zeros(p)))
            if diffBetaAgg[-1]<1e-3:
                break
           
    yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
    yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
     

    sAll.append(s)
    
methodName.append('Proposed method')
sampleSizeUsed.append(np.array(sAll)*2*p)
RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))



fig, ax = plt.subplots()
im = ax.imshow(betaFit)
ax.set_xticks(np.arange(numSystems))
ax.set_xticklabels(systemName)
ax.set_yticks(np.arange(p))
ax.set_yticklabels(betaName)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Variable Selection for Proposed Method")
plt.colorbar(im)
fig.savefig(plotLocation+'Real Data Variable Selection Proposed Method.png', dpi=100)
plt.close()






## IBOSS one system at a time same sample size
yTestAll = np.empty(0)
yHatAll = np.empty(0)
size = np.sum(sampleSizeUsed[0])/numSystems
betaFit = np.zeros([p,numSystems])
sAllOSIBOSS = list()
# s=size/p/2
for k in np.arange(0,numSystems):

    
#    totalDataExtracted = list()
    ## IBOSS Algorithm with beta change
    totalDataToBeReduce = totalTrainData[k]
    xySampled = IBOSSBench(size,totalDataToBeReduce)
    
#    totalDataExtracted.append(xySampled)
    xSampled = xySampled[:,:-1]
    ySampled = xySampled[:,-1]
    
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
        fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                     
    betaFit[:,k] = reg.coef_

        
    yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
    yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
    
    
    sAllOSIBOSS.append(xSampled.shape[0])
    
methodName.append('IBOSS Same Total Sample Size')
sampleSizeUsed.append(np.array(sAllOSIBOSS))

RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))




fig, ax = plt.subplots()
im = ax.imshow(betaFit)
ax.set_xticks(np.arange(numSystems))
ax.set_xticklabels(systemName)
ax.set_yticks(np.arange(p))
ax.set_yticklabels(betaName)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Variable Selection for IBOSS")
plt.colorbar(im)
fig.savefig(plotLocation+'Real Data Variable Selection IBOSS.png', dpi=100)
plt.close()





## random sampling one system at a time same sample size
yTestAll = np.empty(0)
yHatAll = np.empty(0)
sampleSize = int(np.round(size))
betaFit = np.zeros([p,numSystems])
sAllOSrnd = list()

for k in np.arange(0,numSystems):

    totalDataToBeReduce = totalTrainData[k]
    xySampled = RandomSampling(sampleSize,totalDataToBeReduce)
    
    xSampled = xySampled[:,:-1]
    ySampled = xySampled[:,-1]
    
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
        fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                     
    betaFit[:,k] = reg.coef_
    
    yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
    yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
    
    sAllOSrnd.append(xSampled.shape[0])
 
methodName.append('Random Sampling Same Total Sample Size') 
sampleSizeUsed.append(np.array(sAllOSrnd))

RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))




fig, ax = plt.subplots()
im = ax.imshow(betaFit)
ax.set_xticks(np.arange(numSystems))
ax.set_xticklabels(systemName)
ax.set_yticks(np.arange(p))
ax.set_yticklabels(betaName)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Variable Selection for Random Sampling")
plt.colorbar(im)
fig.savefig(plotLocation+'Real Data Variable Selection Random Sampling.png', dpi=100)
plt.close()



## Strat sampling one system at a time same sample size
yTestAll = np.empty(0)
yHatAll = np.empty(0)
sampleSize = int(np.round(size))
betaFit = np.zeros([p,numSystems])
sAllOSstrat = list()

for k in np.arange(0,numSystems):

    totalDataToBeReduce = totalTrainData[k]
    xySampled = StratSampling(sampleSize,totalDataToBeReduce)
    
    xSampled = xySampled[:,:-1]
    ySampled = xySampled[:,-1]
    
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
        fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                     
    betaFit[:,k] = reg.coef_
    
    yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
    yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
    
    sAllOSstrat.append(xSampled.shape[0])
 
methodName.append('Strat Sampling Same Total Sample Size') 
sampleSizeUsed.append(np.array(sAllOSstrat))

RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))



fig, ax = plt.subplots()
im = ax.imshow(betaFit)
ax.set_xticks(np.arange(numSystems))
ax.set_xticklabels(systemName)
ax.set_yticks(np.arange(p))
ax.set_yticklabels(betaName)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Variable Selection for Stratified Sampling")
plt.colorbar(im)
fig.savefig(plotLocation+'Real Data Variable Selection Stratified Sampling.png', dpi=100)
plt.close()






## cluster sampling one system at a time same sample size
yTestAll = np.empty(0)
yHatAll = np.empty(0)
sampleSize = int(np.round(size))
betaFit = np.zeros([p,numSystems])
sAllOSClus = list()

for k in np.arange(0,numSystems):

    totalDataToBeReduce = totalTrainData[k]
    xySampled = ClusterSampling(sampleSize,totalDataToBeReduce)
    
    xSampled = xySampled[:,:-1]
    ySampled = xySampled[:,-1]
    
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
        fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                     
    betaFit[:,k] = reg.coef_
    
    yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
    yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
    
    sAllOSClus.append(xSampled.shape[0])
 
methodName.append('Cluster Sampling Same Total Sample Size') 
sampleSizeUsed.append(np.array(sAllOSClus))


RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))




fig, ax = plt.subplots()
im = ax.imshow(betaFit)
ax.set_xticks(np.arange(numSystems))
ax.set_xticklabels(systemName)
ax.set_yticks(np.arange(p))
ax.set_yticklabels(betaName)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_title("Variable Selection for Cluster Sampling")
plt.colorbar(im)
fig.savefig(plotLocation+'Real Data Variable Selection Cluster Sampling.png', dpi=100)
plt.close()



print(RMSEAll)
print(sampleSizeUsed)

