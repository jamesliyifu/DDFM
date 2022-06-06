# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:15:49 2020

@author: liyifu
"""


# import os

# os.chdir(r"c:\users\liyifu\.julia\conda\3\lib\site-packages")
import statsmodels.api as sm 
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth

from sklearn.metrics import mean_squared_error
from sklearn.metrics import silhouette_score
from math import log
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
import time

def LassoOneSE(xSampled,yResi):
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=10,
     fit_intercept=False,max_iter=10000).fit(xSampled, yResi)
    
    rmse_path_ = np.sqrt(reg.mse_path_)
    # print(rmse_path_)
    rmsePath = np.mean(rmse_path_,axis=1)
    sePath = np.std(rmse_path_,axis=1)/np.sqrt(5)
    minInd = np.argsort(rmsePath)[0]
    maxThreshold = rmsePath[minInd]+sePath[minInd]
    alphaOpt = reg.alphas_[np.where(rmsePath<maxThreshold)[0][0]]
    
    
    reg = Lasso(random_state=0,alpha = alphaOpt,tol=1e-3,
          fit_intercept=False,max_iter=10000).fit(xSampled, yResi)
    return reg
 
def LassoOneSEInt(xSampled,yResi):
    reg = LassoCV(cv=5, random_state=0,tol=1e-3,alphas=[1e1,1e2,1e3,1e4,1e5,1e6],
     fit_intercept=True,max_iter=10000).fit(xSampled, yResi)
    
    rmse_path_ = np.sqrt(reg.mse_path_)
    # print(rmse_path_)
    rmsePath = np.mean(rmse_path_,axis=1)
    sePath = np.std(rmse_path_,axis=1)/np.sqrt(5)
    minInd = np.argsort(rmsePath)[0]
    maxThreshold = rmsePath[minInd]+sePath[minInd]
    alphaOpt = reg.alphas_[np.where(rmsePath<maxThreshold)[0][0]]
    # print(rmsePath)
    # print(maxThreshold)
    # print(alphaOpt)
    reg = Lasso(random_state=0,alpha = alphaOpt,tol=1e-3,
          fit_intercept=True,max_iter=10000).fit(xSampled, yResi)
    return reg
            
def IBOSS(s,totalDataToBeReduce):
    p = totalDataToBeReduce.shape[1]
    xySampled = np.empty([0,p])
    
    for j in range(p-1):
        maxInd = np.argsort(totalDataToBeReduce[:,j])[-s:]
        minInd = np.argsort(totalDataToBeReduce[:,j])[:s]
        
        xySampled = np.vstack((xySampled,totalDataToBeReduce[maxInd,:]))
        xySampled = np.vstack((xySampled,totalDataToBeReduce[minInd,:]))
        
        totalDataToBeReduce = np.delete(totalDataToBeReduce, np.hstack((maxInd,minInd)), 0)
        
    return xySampled
      
def IBOSSBench(size,totalDataToBeReduce):
    p = totalDataToBeReduce.shape[1]
    xySampled = np.empty([0,p])
    s = int(np.floor(size/(p-1)/2))
    
    for j in range(p-1):
        maxInd = np.argsort(totalDataToBeReduce[:,j])[-s:]
        minInd = np.argsort(totalDataToBeReduce[:,j])[:s]
        size -= s*2
        
        xySampled = np.vstack((xySampled,totalDataToBeReduce[maxInd,:]))
        xySampled = np.vstack((xySampled,totalDataToBeReduce[minInd,:]))
        
        totalDataToBeReduce = np.delete(totalDataToBeReduce, np.hstack((maxInd,minInd)), 0)

    if size > 1:
        for j in range(p-1):
            maxInd = np.argsort(totalDataToBeReduce[:,j])[-1:]
            minInd = np.argsort(totalDataToBeReduce[:,j])[:1]
            size -= 2
            
            xySampled = np.vstack((xySampled,totalDataToBeReduce[maxInd,:]))
            xySampled = np.vstack((xySampled,totalDataToBeReduce[minInd,:]))
            
            totalDataToBeReduce = np.delete(totalDataToBeReduce, np.hstack((maxInd,minInd)), 0)
                    
            if size <= 0:
                
                break
        
    return xySampled

def ClusterSampling(sampleSize,totalDataToBeReduce):

    range_n_clusters = [2]
    
    silhouette_avg_All = list()
    cluster_labels_All = list()
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels_All.append(clusterer.fit_predict(totalDataToBeReduce[:,:-2]))
        silhouette_avg_All.append(silhouette_score(totalDataToBeReduce[:,:-2], cluster_labels_All[-1]))
        
    bestLabels = cluster_labels_All[np.where(silhouette_avg_All==np.max(silhouette_avg_All))[0][0]]
        
    
    xySampled = np.empty([0,totalDataToBeReduce.shape[1]])
    for i in range(np.max(bestLabels)+1):
        currSampleSize = np.max([int(np.round(sampleSize*(np.sum(bestLabels==i)/totalDataToBeReduce.shape[0]))),1])
        # print([i,currSampleSize])
        xySampled = np.vstack((xySampled, RandomSampling(currSampleSize,totalDataToBeReduce[bestLabels==i,:])))
        
    return xySampled

def CloudClusterSampling(totalData):
    totalDataSum = np.empty([0,totalData.shape[1]])
    Alllabel = np.empty(0)
    for i in range(totalData.shape[2]):
        totalDataSum = np.vstack((totalDataSum, totalData[:,:,i]))
        Alllabel = np.hstack((Alllabel, np.repeat(i, totalData[:,:,i].shape[0])))
        
    range_n_clusters = [2]
    
    silhouette_avg_All = list()
    cluster_labels_All = list()
    
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels_All.append(clusterer.fit_predict(totalDataSum[:,:-2]))
        silhouette_avg_All.append(silhouette_score(totalDataSum[:,:-2], cluster_labels_All[-1]))
        
    bestLabels = cluster_labels_All[np.where(silhouette_avg_All==np.max(silhouette_avg_All))[0][0]]
    
    return [totalDataSum,Alllabel,bestLabels]


def RandomSampling(sampleSize,totalDataToBeReduce):
    
    randomInd = np.arange(len(totalDataToBeReduce))
    np.random.shuffle(randomInd)
    
    dataFiltered = totalDataToBeReduce[randomInd[:sampleSize],:]
    
    return dataFiltered

def StratSampling(sampleSize,totalDataToBeReduce):
    
    stratifiedInd = np.linspace(0,len(totalDataToBeReduce)-1,sampleSize,dtype=int)
    
    dataFiltered = totalDataToBeReduce[stratifiedInd,:]
    
    return dataFiltered

        
def FiveFoldCV(x,y):
    p = x.shape[1]
    
    mseAll = list()
    l1List = [1e-6,1,1e1,1e2,1e3,1e4,1e5,1e6,1e7]
    kf = KFold(n_splits=5) 
    
    for l1 in l1List:
        
        errorAll = np.empty(0)
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            betaResiFit=np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train)+l1*np.diag(np.ones(p))),X_train.T),y_train)
            errorAll = np.hstack((errorAll,y_test-np.dot(X_test, betaResiFit)))
            
        mseAll.append(mean_squared_error(errorAll,np.zeros(len(errorAll))))
        
    bestInd = np.argsort(mseAll)[0] 
    betaResiFit=np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)+l1List[bestInd]*np.diag(np.ones(p))),x.T),y)   
              
    return betaResiFit        


def Network_IBOSS_and_Benchmarks(n, similarity, modelSparsity, signal_to_noise_ratio, totalNumIte):

    p = 20
    sigmaInter = 0.3
    numMethods = 5
    numSystems = 10
    nAlpha=50
    
    sampleSizeUsedSum = np.zeros([numMethods,numSystems])
    RMSEAllSum = list()
    RMSEPredAllSum = list()
    TimeAllSum = list()
    
    for rep in range(totalNumIte):

        print([rep, n, similarity, modelSparsity, signal_to_noise_ratio])
    
        
        totalData = np.empty([n,p+2,numSystems])
        totalTestData = np.empty([int(n+n/9*1)-n,p+2,numSystems])
        
        trueBeta = np.zeros([p+1,numSystems])
        
        ## proposed method with beta change
        
        #V = np.random.rand(p,p)
        #
        #betaPool = orth(V)[:p,:]
        #betaRandInd = np.arange(p)
        diffBetaAgg = list()
        baseBeta = np.random.normal(0, 1, p+1)
        RMSEAll = list()
        RMSEPredAll = list()
        TimeAll = list()
        zeroInd = np.arange(p)
        np.random.shuffle(zeroInd)
        
        
        
        # Two_Beta = list()
        
        # generate the simulation data and the true model
        for currSys in range(numSystems):
            
            trueBeta[:,currSys] += (baseBeta+np.sign(baseBeta)*np.random.rand(p+1)*(1/similarity))
            trueBeta[zeroInd[:int(np.round((1-modelSparsity)*p))],currSys] = 0
            mu = np.random.normal(0, 1, p)
            sigma = np.diag(np.ones(p)) # mean and standard deviation
            sigma[sigma==0]=sigmaInter
            
            x = np.hstack((np.random.multivariate_normal(mu, sigma, int(n+n/9*1)),np.ones(int(n+n/9*1)).reshape(-1,1)))
            # print(x.shape)
            yraw = np.dot(x,trueBeta[:,currSys]).reshape(-1,1)
            noise = np.random.normal(0, 1, int(n+n/9*1)).reshape(-1,1)
            
            noiseMultiplier=np.sqrt(np.var(yraw)/(signal_to_noise_ratio*np.var(noise)))
            
            y = yraw + noiseMultiplier*noise
            
            # meanTrainY = np.mean(y[:n])
            
            totalData[:,:,currSys]=np.hstack((x[:n,:],y[:n]))
            totalTestData[:,:,currSys]=np.hstack((x[n:,:],y[n:]))
            
        
     
        
        ################Sim Run###################

        
        
        
        

        methodName = list()
        
        #### Full Data

        sampleSizeUsed= list()
        
        betaFitFull = np.zeros([p+1,numSystems])
        sAllOSrnd = list()
        timePerEpoch = list()
        timePerEpoch.append(0.0)
        
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
                
        for k in np.arange(0,numSystems):
        
            xSampled = totalData[:,:-1,k]
            ySampled = totalData[:,-1,k]
            
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                          fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                         
            betaFitFull[:,k] = reg.coef_
            
            betaFitFull[-1,k] = np.mean(totalData[:,-1,k])-np.dot(np.mean(totalData[:,:-2,k],axis=0),betaFitFull[:p,k])
            
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)
            
            yTestAll = np.hstack((yTestAll,totalTestData[:,-1,k]))
            yHatAll = np.hstack((yHatAll,np.dot(totalTestData[:,:-1,k],betaFitFull[:,k])))
            sAllOSrnd.append(xSampled.shape[0])
         
        methodName.append('Full Data') 
        RMSEAll.append(np.sqrt(mean_squared_error(betaFitFull, trueBeta)))
        RMSEPredAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        sampleSizeUsed.append(np.array(sAllOSrnd))
        TimeAll.append(timePerEpoch)
        
        
        #### IBOSS Network
        timePerEpoch = list()
 
        
        
        currP = totalData[:,:,0].shape[1]-1
        AggX = np.empty([0,currP*(numSystems+1)])
        Aggy = np.empty(0)

        s= 1
        sAllPre = list()
        for k in np.arange(0,numSystems):
            ## IBOSS Algorithm with beta change
            totalDataToBeReduce = totalData[:,:,k]
            xySampled = IBOSS(s,totalDataToBeReduce)
        
            currN = xySampled.shape[0]
            
            currX = np.zeros([currN,currP*(numSystems+1)])
            currX[:,0:currP] = xySampled[:,:currP]
            currX[:,(k+1)*currP:(k+2)*currP] = xySampled[:,:currP]*1/numSystems
            
            AggX = np.vstack((AggX,currX))
            Aggy = np.hstack((Aggy,xySampled[:,-1]))
            
        t = time.time()    
        reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                      fit_intercept=False,max_iter=10000).fit(AggX, Aggy)
        
        # time it takes for the general model

        benchBeta = reg.coef_[:currP]  
        
        eachElapsed = time.time() - t   
        timePerEpoch.append(eachElapsed)
        
        sAllPre.append(np.repeat(s,numSystems))         
                 
        
        
        maxN = int(int(n+n/7*3)/2/p)
        betaFitProp = np.zeros([p+1,numSystems])
        betaFitPre = np.zeros([p+1,numSystems])
        diffBetaAgg = list()
        sAll = list()

        
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        
                
        for k in np.arange(0,numSystems):
            diffBetaAgg = list()
            eachElapsed = 0

            for s in np.arange(1,maxN):
            #    totalDataExtracted = list()
                ## IBOSS Algorithm with beta change
                totalDataToBeReduce = totalData[:,:,k]
                xySampled = IBOSS(s,totalDataToBeReduce)
                
            #    totalDataExtracted.append(xySampled)
                xSampled = xySampled[:,:-1]
                ySampled = xySampled[:,-1]
                yResi = ySampled-np.dot(xSampled,benchBeta)
                # reg = LassoOneSE(xSampled,yResi)
                t = time.time()
                
                reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                              fit_intercept=False,max_iter=10000).fit(xSampled, yResi)
                             
                eachElapsed += time.time() - t
                
                
                betaAdd = reg.coef_
                # betaResiFit = FiveFoldCV(xSampled,yResi)
                
                betaFitPre[:p,k] = betaFitProp[:p,k]
                betaFitProp[:p,k] = benchBeta[:p]+betaAdd[:p]
                
                diffBeta = mean_squared_error(betaFitPre[:p,k],betaFitProp[:p,k])
                if s >1:
                    diffBetaAgg.append(diffBeta/mean_squared_error(betaFitPre[:p,k],np.zeros(p)))
                    if diffBetaAgg[-1]<1e-3:
                        break
            
            
            betaFitProp[-1,k] = np.mean(totalData[:,-1,k])-np.dot(np.mean(totalData[:,:-2,k],axis=0),betaFitProp[:p,k])  
            
            # time for each

            timePerEpoch.append(eachElapsed)
            
            
            yTestAll = np.hstack((yTestAll,totalTestData[:,-1,k]))
            yHatAll = np.hstack((yHatAll,np.dot(totalTestData[:,:-1,k],betaFitProp[:,k])))
            sAll.append(s)
            
        methodName.append('Proposed method')
        RMSEAll.append(np.sqrt(mean_squared_error(betaFitProp, trueBeta)))
        RMSEPredAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        TimeAll.append(timePerEpoch)
          
        
        sAll = np.max(np.hstack((np.array(sAllPre).reshape(-1,1),np.array(sAll).reshape(-1,1))),axis=1).reshape(-1)
        sampleSizeUsed.append(np.array(sAll)*2*p)
        
        
        
        
        
        
        #### IBOSS one system at a time same sample size
        size = np.sum(sampleSizeUsed[1])/numSystems
        betaFitIBOSS = np.zeros([p+1,numSystems])
        sAllOSIBOSS = list()
        
        
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        
        
        timePerEpoch = list()
        timePerEpoch.append(0.0)
        
        for k in np.arange(0,numSystems):
            
            ## IBOSS Algorithm with beta change
            totalDataToBeReduce = totalData[:,:,k]
            xySampled = IBOSSBench(size,totalDataToBeReduce)
        
            xSampled = xySampled[:,:-1]
            ySampled = xySampled[:,-1]
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                          fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                         
            betaFitIBOSS[:,k] = reg.coef_
            
            betaFitIBOSS[-1,k] = np.mean(totalData[:,-1,k])-np.dot(np.mean(totalData[:,:-2,k],axis=0),betaFitIBOSS[:p,k]) 
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)
            
            yTestAll = np.hstack((yTestAll,totalTestData[:,-1,k]))
            yHatAll = np.hstack((yHatAll,np.dot(totalTestData[:,:-1,k],betaFitIBOSS[:,k])))
            sAllOSIBOSS.append(xSampled.shape[0])
        
        methodName.append('IBOSS Same Total Sample Size')    
        RMSEAll.append(np.sqrt(mean_squared_error(betaFitIBOSS, trueBeta)))
        RMSEPredAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        sampleSizeUsed.append(np.array(sAllOSIBOSS))
        TimeAll.append(timePerEpoch)
        
        
        
        #### random sampling one system at a time same sample size
        sampleSize = int(np.round(size))
        betaFitRandom = np.zeros([p+1,numSystems])
        sAllOSrnd = list()
        timePerEpoch = list()
        timePerEpoch.append(0.0)
        
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
                
        for k in np.arange(0,numSystems):
        
            
            totalDataToBeReduce = totalData[:,:,k]
            xySampled = RandomSampling(sampleSize,totalDataToBeReduce)
            
            xSampled = xySampled[:,:-1]
            ySampled = xySampled[:,-1]
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                          fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                         
            betaFitRandom[:,k] = reg.coef_
            
            betaFitRandom[-1,k] = np.mean(totalData[:,-1,k])-np.dot(np.mean(totalData[:,:-2,k],axis=0),betaFitRandom[:p,k])
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)
            
            yTestAll = np.hstack((yTestAll,totalTestData[:,-1,k]))
            yHatAll = np.hstack((yHatAll,np.dot(totalTestData[:,:-1,k],betaFitRandom[:,k])))
            sAllOSrnd.append(xSampled.shape[0])
         
        methodName.append('Random Sampling Same Total Sample Size') 
        RMSEAll.append(np.sqrt(mean_squared_error(betaFitRandom, trueBeta)))
        RMSEPredAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        sampleSizeUsed.append(np.array(sAllOSrnd))
        TimeAll.append(timePerEpoch)
        
        
        
        #### Strat sampling one system at a time same sample size
        betaFitStrat = np.zeros([p+1,numSystems])
        sAllOSstrat = list()
        timePerEpoch = list()
        timePerEpoch.append(0.0)
        
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
                       
        for k in np.arange(0,numSystems):
            
            
            totalDataToBeReduce = totalData[:,:,k]
            xySampled = StratSampling(sampleSize,totalDataToBeReduce)
            
            xSampled = xySampled[:,:-1]
            ySampled = xySampled[:,-1]
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                          fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                         
            betaFitStrat[:,k] = reg.coef_
            
            betaFitStrat[-1,k] = np.mean(totalData[:,-1,k])-np.dot(np.mean(totalData[:,:-2,k],axis=0),betaFitStrat[:p,k])       
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)
            
            yTestAll = np.hstack((yTestAll,totalTestData[:,-1,k]))
            yHatAll = np.hstack((yHatAll,np.dot(totalTestData[:,:-1,k],betaFitStrat[:,k])))
            sAllOSstrat.append(xSampled.shape[0])
         
        methodName.append('Strat Sampling Same Total Sample Size') 
        RMSEAll.append(np.sqrt(mean_squared_error(betaFitStrat, trueBeta)))
        RMSEPredAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        sampleSizeUsed.append(np.array(sAllOSstrat))
        TimeAll.append(timePerEpoch)
        
        

        
        
        sampleSizeUsedSum += np.array(sampleSizeUsed)
        RMSEAllSum.append(RMSEAll)
        RMSEPredAllSum.append(RMSEPredAll)
        TimeAllSum.append(TimeAll)
        
        print(sampleSizeUsed)
        print(np.mean(np.array(RMSEAllSum), axis=0))
        print(np.mean(np.array(RMSEPredAllSum), axis=0))
        print(np.mean(np.array(TimeAllSum), axis=0))
        
    sampleSizeUsedAvg = sampleSizeUsedSum/totalNumIte  
    RMSEAllSum = np.array(RMSEAllSum)
    
    RMSEPredAllSum = np.array(RMSEPredAllSum)
    
    TimeAllSum = np.array(TimeAllSum)
    
    print(np.mean(RMSEAllSum, axis=0))
    print(np.mean(RMSEPredAllSum, axis=0))
    
    
    return [sampleSizeUsedAvg, 
            np.mean(RMSEAllSum, axis=0), np.std(RMSEAllSum, axis=0)/np.sqrt(totalNumIte),
            np.mean(RMSEPredAllSum, axis=0), np.std(RMSEPredAllSum, axis=0)/np.sqrt(totalNumIte),
            TimeAllSum]




def Network_RealData_DistN(outfile, furnance_index, totalNumIte):
    trteP = 0.9
    totalVar = 56
    nAlpha = 50
    indDel = [1,3,4,5,6,7,8,9,11,12,13,15,16,17,18,19,21,23,24,27,28,29,30,31,33,34,37,38,39,40,42,43,45,47,49,50,51,52]
    numMethods = 5
    # num_trancated = 500
    # numSampleUsedPerSystem = 1
    TimeAllAgg = list()
    RMSEAllAgg = list()
    BetaAllAgg = list()
    
    sampleSizeUsedSum = np.zeros([numMethods,len(furnance_index)])
    
    for rep in range(totalNumIte):
        totalTrainData = list()
        totalTestData = list()
    
        
        for i in range(len(furnance_index)):
            currData = np.load(outfile+'\\'+str(furnance_index[i])+'_data.npy',allow_pickle=True)
        
            xTrainAll = np.empty([0,totalVar-len(indDel)+1])
            xTestAll = np.empty([0,totalVar-len(indDel)+1])
            yTrainAll = np.empty(0)
            yTestAll = np.empty(0)   
            
            
            for j in range(len(currData)):
        
            # for j in range(1):    
                dataImported = currData[j][:,:totalVar]
                if dataImported.shape[0]<1000:
                    continue
                
                y = np.array(dataImported[:,34],dtype=float)[1:]
                x = np.array(np.delete(dataImported, indDel, axis=1),dtype=float)[1:,:]
                n = x.shape[0]
                x = np.hstack((x,np.ones(n).reshape(-1,1)))
                
                
                shuffleInd = np.arange(n)
                
                np.random.shuffle(shuffleInd)
                
                xTrain, xTest = x[shuffleInd[:int(n*trteP)],:], x[shuffleInd[int(n*trteP):],:]
                yTrain, yTest = y[shuffleInd[:int(n*trteP)]], y[shuffleInd[int(n*trteP):]]
                
                xTrainAll = np.vstack((xTrainAll,xTrain))
                # scaler = preprocessing.StandardScaler().fit(xTrainAll)
                xTestAll = np.vstack((xTestAll,xTest))
                
                # xTrainAll = scaler.transform(xTrainAll)
                # xTestAll = scaler.transform(xTestAll)
                
                yTrainAll = np.hstack((yTrainAll,yTrain))
                yTestAll = np.hstack((yTestAll,yTest))
                
            yTestAll = yTestAll-np.mean(yTrainAll)
            yTrainAll = yTrainAll-np.mean(yTrainAll)
            
            totalTrainData.append(np.hstack((xTrainAll,
                                             yTrainAll.reshape(-1,1))))
            
            totalTestData.append(np.hstack((xTestAll,
                                            yTestAll.reshape(-1,1))))
                            
                    
        
        #############Real Run######################
        BetaAll = list()
        TimeAll = list()   
        RMSEAll = list()
        sampleSizeUsed = list()
        
        methodName = list()        
            
        p = totalTrainData[0].shape[1]-1
        numSystems = len(totalTrainData)
        print(numSystems)
        
                
        #### All Data

        timePerEpoch = list()
        timePerEpoch.append(0.0)
        
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        # sampleSize = int(np.round(size))
        betaFit = np.zeros([p,numSystems])
        sAllOSstrat = list()
        
        for k in np.arange(0,numSystems):

            
            xSampled = totalTrainData[k][:,:-1]
            ySampled = totalTrainData[k][:,-1]
            t = time.time()
            
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                             
            betaFit[:,k] = reg.coef_
            
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)
            
            betaFit[-1,k]= np.mean(totalTrainData[k][:,-1])-np.dot(np.mean(totalTrainData[k][:,:-2],axis=0),betaFit[:-1,k]) 

          
            yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
            yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
            
            sAllOSstrat.append(xSampled.shape[0])
         
        methodName.append('All Sample Size') 

        sampleSizeUsed.append(np.array(sAllOSstrat))
        
        RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        TimeAll.append(timePerEpoch)
        BetaAll.append(np.array(betaFit))
        
        
        
        #### IBOSS Network  
        timePerEpoch = list()

        AggX = np.empty([0,p*(numSystems+1)])
        Aggy = np.empty(0)
        
        
        s= 1
        sAllPre = list()
        for k in np.arange(0,numSystems):
            ## IBOSS Algorithm with beta change
            totalDataToBeReduce = totalTrainData[k]
            xySampled = IBOSS(s,totalDataToBeReduce)
        
            currN = xySampled.shape[0]
            
            currX = np.zeros([currN,p*(numSystems+1)])
            currX[:,0:p] = xySampled[:,:p]
            currX[:,(k+1)*p:(k+2)*p] = xySampled[:,:p]*1/numSystems
            
            AggX = np.vstack((AggX,currX))
            Aggy = np.hstack((Aggy,xySampled[:,-1]))
            
        t = time.time()    
        reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                      fit_intercept=False,max_iter=10000).fit(AggX, Aggy)
        
        eachElapsed = time.time() - t
        timePerEpoch.append(eachElapsed)  
        
        benchBeta = reg.coef_[:p]  
        
        sAllPre.append(np.repeat(s,numSystems))         
                 

        
    
        betaFit = np.zeros([p,numSystems])
        betaFitPre = np.zeros([p,numSystems])

        diffBetaAgg = list()

        sAll = list()
        methodName = list()
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        

            
        for k in np.arange(0,numSystems):
            eachElapsed = 0
            
            diffBetaAgg = list()
            n = totalTrainData[k].shape[0]
            maxN = int(n/2/p)
            totalDataToBeReduce = totalTrainData[k]
                
            for s in np.arange(1,maxN):
            #    totalDataExtracted = list()
                ## IBOSS Algorithm with beta change
        
            
                xySampled = IBOSS(s,totalDataToBeReduce)
                
            #    totalDataExtracted.append(xySampled)
                xSampled = xySampled[:,:-1]
                ySampled = xySampled[:,-1]
                yResi = ySampled-np.dot(xSampled,benchBeta)
                
                # betaAdd = FiveFoldCV(xSampled,yResi)
                t = time.time()
                reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                              fit_intercept=False,max_iter=10000).fit(xSampled, yResi)
                eachElapsed += time.time() - t
                
                betaAdd = reg.coef_
        
                betaFitPre[:,k] = betaFit[:,k]
                betaFit[:,k] = benchBeta+betaAdd
                
        
                diffBeta = mean_squared_error(betaFitPre[:,k],betaFit[:,k])
                if s >1:
                    diffBetaAgg.append(diffBeta/mean_squared_error(betaFitPre[:,k],np.zeros(p)))
                    if diffBetaAgg[-1]<1e-3:
                        break
                    
            betaFit[-1,k]= np.mean(totalDataToBeReduce[:,-1])-np.dot(np.mean(totalDataToBeReduce[:,:-2],axis=0),betaFit[:-1,k]) 

            yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
            yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 

            timePerEpoch.append(eachElapsed)
        
            sAll.append(s)
            
        methodName.append('Proposed method')
        
        sAll = np.max(np.hstack((np.array(sAllPre).reshape(-1,1),np.array(sAll).reshape(-1,1))),axis=1).reshape(-1)
        sampleSizeUsed.append(np.array(sAll)*2*p)
        RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        TimeAll.append(timePerEpoch)
        BetaAll.append(np.array(betaFit))
        
        # trtrue = totalTestData[k][:,-1]
        # trhat = np.dot(totalTestData[k][:,:-1],betaFit[:,k])
        
        # resi = trhat-trtrue
        
        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].scatter(trhat, resi,s=2) #row=0, col=0
        # ax[1, 0].hist(resi) #row=1, col=0
        # ax[0, 1].scatter(resi[:-1],resi[1:],s=2) #row=1, col=1
        # # ax[0, 1].plot()  #row=0, col=1
        # res = stats.probplot(resi, plot=plt)
        # plt.tight_layout()
        # fig.show()
                
            
            
        
        
        
        totalSize = 0
        for systemInd in range(len(totalTrainData)):
            totalSize += totalTrainData[systemInd].shape[0]
            
            
            
            
        ## IBOSS one system at a time same sample size
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        timePerEpoch = list()
        timePerEpoch.append(0.0)

        
        # size = np.sum(sampleSizeUsed[0])/numSystems
        betaFit = np.zeros([p,numSystems])
        sAllOSIBOSS = list()
        
        # s=size/p/2
        for k in np.arange(0,numSystems):
            
            size = int(np.sum(sampleSizeUsed[1])*totalTrainData[k].shape[0]/totalSize)
        #    totalDataExtracted = list()
            ## IBOSS Algorithm with beta change
            totalDataToBeReduce = totalTrainData[k]
            xySampled = IBOSSBench(size,totalDataToBeReduce)
            
        #    totalDataExtracted.append(xySampled)
            xSampled = xySampled[:,:-1]
            ySampled = xySampled[:,-1]
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                             
            betaFit[:,k] = reg.coef_
        
            betaFit[-1,k]= np.mean(totalDataToBeReduce[:,-1])-np.dot(np.mean(totalDataToBeReduce[:,:-2],axis=0),betaFit[:-1,k]) 
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)

            
            
            yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
            yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
            
            
            sAllOSIBOSS.append(xSampled.shape[0])
            
        methodName.append('IBOSS Same Total Sample Size')
        sampleSizeUsed.append(np.array(sAllOSIBOSS))
        
        RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        TimeAll.append(timePerEpoch)
        BetaAll.append(np.array(betaFit))
        
        ## random sampling one system at a time same sample size
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        # sampleSize = int(np.round(size))
        betaFit = np.zeros([p,numSystems])
        sAllOSrnd = list()
        timePerEpoch = list()
        timePerEpoch.append(0.0)

        
        for k in np.arange(0,numSystems):
            
            size = int(np.sum(sampleSizeUsed[1])*totalTrainData[k].shape[0]/totalSize)
            totalDataToBeReduce = totalTrainData[k]
            xySampled = RandomSampling(size,totalDataToBeReduce)
            
            xSampled = xySampled[:,:-1]
            ySampled = xySampled[:,-1]
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                             
            betaFit[:,k] = reg.coef_
            
            betaFit[-1,k]= np.mean(totalDataToBeReduce[:,-1])-np.dot(np.mean(totalDataToBeReduce[:,:-2],axis=0),betaFit[:-1,k]) 
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)

        
            yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
            yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
            
            sAllOSrnd.append(xSampled.shape[0])
         
        methodName.append('Random Sampling Same Total Sample Size') 
        sampleSizeUsed.append(np.array(sAllOSrnd))
        
        RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        TimeAll.append(timePerEpoch)
        BetaAll.append(np.array(betaFit))
        
        ## Strat sampling one system at a time same sample size
        yTestAll = np.empty(0)
        yHatAll = np.empty(0)
        # sampleSize = int(np.round(size))
        betaFit = np.zeros([p,numSystems])
        sAllOSstrat = list()
        timePerEpoch = list()
        timePerEpoch.append(0.0)
        betaFitPerEpoch = list()
        
        for k in np.arange(0,numSystems):
            
            size = int(np.sum(sampleSizeUsed[1])*totalTrainData[k].shape[0]/totalSize)
            totalDataToBeReduce = totalTrainData[k]
            xySampled = StratSampling(size,totalDataToBeReduce)
            
            xSampled = xySampled[:,:-1]
            ySampled = xySampled[:,-1]
            t = time.time()
            reg = LassoCV(cv=5, random_state=0,tol=1e-3,n_alphas=nAlpha,
                fit_intercept=False,max_iter=10000).fit(xSampled, ySampled)
                             
            betaFit[:,k] = reg.coef_
            
            betaFit[-1,k]= np.mean(totalDataToBeReduce[:,-1])-np.dot(np.mean(totalDataToBeReduce[:,:-2],axis=0),betaFit[:-1,k]) 
            eachElapsed = time.time() - t
            timePerEpoch.append(eachElapsed)
            betaFitPerEpoch.append(betaFit)
            
            yTestAll = np.hstack((yTestAll, totalTestData[k][:,-1])) 
            yHatAll = np.hstack((yHatAll, np.dot(totalTestData[k][:,:-1],betaFit[:,k]))) 
            
            sAllOSstrat.append(xSampled.shape[0])
         
        methodName.append('Strat Sampling Same Total Sample Size') 
        sampleSizeUsed.append(np.array(sAllOSstrat))
        
        RMSEAll.append(np.sqrt(mean_squared_error(yTestAll,yHatAll)))
        TimeAll.append(timePerEpoch)
        BetaAll.append(np.array(betaFit))
        

        
        sampleSizeUsedSum += np.array(sampleSizeUsed)
        RMSEAllAgg.append(RMSEAll)
        TimeAllAgg.append(TimeAll)
        BetaAllAgg.append(np.array(BetaAll))
        
        print(np.mean(np.array(RMSEAllAgg), axis=0))
        print(np.array(BetaAllAgg).shape)
        # print(sampleSizeUsedSum)

    sampleSizeUsedAvg = sampleSizeUsedSum/totalNumIte  
    TimeAllAgg = np.array(TimeAllAgg)        
    BetaAllAgg = np.array(BetaAllAgg)     
        
    return [sampleSizeUsedAvg, np.mean(RMSEAllAgg, axis=0), np.std(RMSEAllAgg, axis=0)/np.sqrt(totalNumIte),TimeAllAgg, BetaAllAgg]



