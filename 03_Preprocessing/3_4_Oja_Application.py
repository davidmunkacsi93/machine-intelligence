#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:15:21 2017

@author: fabian
"""

# 3.4 1)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import matplotlib.cm as cm

dataSet = np.loadtxt('data-onlinePCA.txt', delimiter=',', 
                     #dtype={'names': ('time','V1', 'V2'), 
                    #'formats': ('str', 'f4', 'f4')}, 
                    skiprows=1, 
                    usecols=(1,2))

#print(dataSet)
dataBlocks = dataSet.reshape(10,200,2)
#print(dataBlocks)
#print(dataBlocks.shape)
dataMean = np.mean(dataBlocks, axis=1)
print(dataMean.shape)

title = 'Scatterplot of dataset'
#plt.figure()#figsize=(20,7))
#for i in range(0, dataBlocks.shape[0]):
#    plt.scatter(dataBlocks[i,:,0], dataBlocks[i,:,1], s=2)
#plt.show

plt.figure()#figsize=(20,7))
for i in range(0, dataBlocks.shape[0]):
    batchPCA = PCA(dataBlocks[i], standardize=False)
    plt.scatter(dataBlocks[i,:,0], dataBlocks[i,:,1], s=2)
    plt.arrow(dataMean[i,0], dataMean[i,1], batchPCA.Wt[0,0], batchPCA.Wt[0,1], 
         head_width=0, head_length=0)

plt.show



#w = np.array([-1, 1]) # initialize w
eps = [0.002, 0.04, 0.45]

for e in eps:
    plt.figure(figsize=(20,7))
    wArray = np.zeros((2000,2))
    w = np.array([dataMean[0,0], dataMean[0,1]]) 
    
    
    for i in range(0, dataSet.shape[0]):
        x = dataSet[i]
        y = np.dot(w.T, x)
        dW = e*y*(x-y*w)
        w = w + dW
        wArray[i] = w
    
    wArray = wArray.reshape(10,200,2)
    colors = cm.rainbow(np.linspace(0, 1, 10))
    for i in range(0, dataBlocks.shape[0]):
        plt.scatter(dataBlocks[i,:,0], dataBlocks[i,:,1], s=2, color=colors[i])
        plt.plot(wArray[i,:,0], wArray[i,:,1], color=colors[i], linestyle='-')
        plt.scatter(wArray[i,:,0], wArray[i,:,1], s=10, facecolor=colors[i], edgecolor='k')
        
        #print(x, y, dW, w)
    print(e)
    plt.show