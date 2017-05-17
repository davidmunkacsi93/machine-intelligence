#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:15:21 2017

@author: fabian
"""

### 3.4

## 1) Make a scatter plot of the data and indicate the time index by the color of the datapoints

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import matplotlib.cm as cm

dataSet = np.loadtxt('./data-onlinePCA.txt', delimiter=',',
                    skiprows=1, 
                    usecols=(1,2))

# divide the data into 10 sets covering each one second
dataBlocks = dataSet.reshape(10,200,2)
# get the mean of each set (in both dimensions)
dataMean = np.mean(dataBlocks, axis=1)


plt.figure(figsize=(20,7))
for i in range(0, dataBlocks.shape[0]):
    strLabel = 'sec: ' + str(i) + '-' + str(i+1)
    plt.scatter(dataBlocks[i,:,0], dataBlocks[i,:,1], s=2, label=strLabel)
plt.legend()
plt.title('Scatterplot of dataset in  10 blocks')
plt.xlabel('x dimension')
plt.ylabel('y dimension')
plt.show()


## 2) Determine the principal components (using batch PCA) and plot the first PC for each of the 10 blocks in the same plot as the original data
colors = cm.rainbow(np.linspace(0, 1, 10))

plt.figure(figsize=(20,7))
for i in range(0, dataBlocks.shape[0]):
    strLabel = 'sec: ' + str(i) + '-' + str(i+1)
    batchPCA = PCA(dataBlocks[i], standardize=False)                #calc the PCA
    plt.scatter(dataBlocks[i,:,0], dataBlocks[i,:,1], s=2, label=strLabel, color=colors[i])
    plt.arrow(dataMean[i,0], dataMean[i,1], batchPCA.Wt[0,0], batchPCA.Wt[0,1], head_width=0, head_length=0, color=colors[i])    #plot the first eigenvector
plt.legend()
plt.title('Scatterplot of dataset in  10 blocks with 1st PC as arrow')
plt.xlabel('x dimension')
plt.ylabel('y dimension')
plt.show()

## 3) Implement Oja’s rule and apply it with a learning-rate parameter ε ∈ {0.002, 0.04, 0.45} to the dataset

eps = [0.002, 0.04, 0.45]
for e in eps:
    
    #init required arrays
    plt.figure(figsize=(20,7))
    wArray = np.zeros((2000,2))
    
    #we initialize w with the mean of the first datablock - could be initialized with other values
    w = np.array([dataMean[0,0], dataMean[0,1]]) 
    
    # for each datapoint apply Oja's rule successively
    for i in range(0, dataSet.shape[0]):
        x = dataSet[i]
        y = np.dot(w.T, x)
        dW = e*y*(x-y*w)
        w = w + dW
        wArray[i] = w
    
    #plot the weights and show their development
    wArray = wArray.reshape(10,200,2)
    for i in range(0, dataBlocks.shape[0]):
        plt.scatter(dataBlocks[i,:,0], dataBlocks[i,:,1], s=2, color=colors[i])
        plt.plot(wArray[i,:,0], wArray[i,:,1], color=colors[i], linestyle='-')
        plt.scatter(wArray[i,:,0], wArray[i,:,1], s=10, facecolor=colors[i], edgecolor='k')
    plt.title('Scatterplot of dataset with weight development at eps=' + str(e))
    plt.xlabel('x dimension')
    plt.ylabel('y dimension')
    plt.show()
