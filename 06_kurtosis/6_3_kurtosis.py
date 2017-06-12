import numpy as np
import scipy as scipy
import scipy.io as io
from sklearn.decomposition import PCA
from sklearn import preprocessing as pre
import math
import matplotlib.pyplot as plt
import seaborn as sea
import pandas as pd

#load data from mat file
distrib = io.loadmat("distrib.mat")

normal = distrib.get("normal")
uniform = distrib.get("uniform")
laplacian = distrib.get("laplacian")

# a) apply mixing
def mix(x):
    A = np.asarray([[4,3],[2,1]])
    return np.dot(A,x)

# b) center to zero mean 
def center(x):
    return x - x.mean(axis=1, keepdims=True)

# c) apply PCA and project data onto PC
def applyPCA(x):
    pca = PCA()
    pca.fit(x)
    return pca.transform(x)

# d) scale data to unit variance
def scale(x):
    return pre.maxabs_scale(x)

# e rotate data
def rotate(x, angle): # x of shape (2,n)
    R = np.asarray([[math.cos(angle), -math.sin(angle)],[math.sin(angle), math.cos(angle)]])
    return np.dot(R, x)

def rotateAndGetKurtosis(x, angle):
    scaledRot = rotate(x.T, angle).T
    kurt = scipy.stats.kurtosis(scaledRot)    
    #print("kurt: ", kurt)
    return kurt

def findMinAndMaxKurt(x, angles):
    kurts = np.empty((angles.shape[0],2))
    idx = 0
    for angle in angles:
        kurt = rotateAndGetKurtosis(scaledNormal, angle)
        kurts[idx] = kurt
        idx += 1
    
    maxIdx = np.argmax(kurts[:,0])
    minIdx = np.argmin(kurts[:,1])
    return maxIdx, minIdx

def plotdata(source, mixed, centered, projected, scaled, rotMin, rotMax):    
    # Plotting the results.
 
    df = pd.DataFrame(source, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Sources')
    
    df = pd.DataFrame(mixed, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Mixed')
    
    df = pd.DataFrame(centered, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Centered')
    
    df = pd.DataFrame(projected, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Projected')
    
    df = pd.DataFrame(scaled, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Scaled')
    
    df = pd.DataFrame(rotMin, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Rotation min kurtosis')
    
    df = pd.DataFrame(rotMax, columns=["x", "y"])
    sea.jointplot(x="x", y="y", data=df)
    plt.title('Rotation max kurtosis')

def runExercise(x):
    xNormal = mix(x)
    xNormalCentered = center(xNormal)
    # transpose data for further processing
    xNormalCentered = xNormalCentered.T
    
    projectedNormalCentered = applyPCA(xNormalCentered)
    scaledNormal = scale(projectedNormalCentered)
    
    angles = np.arange(0, 2, 1/50)
    angles = angles * math.pi
    maxIdx, minIdx = findMinAndMaxKurt(scaledNormal, angles)
    
    rotNormMax = rotate(scaledNormal.T, angles[maxIdx])
    rotNormMin = rotate(scaledNormal.T, angles[minIdx])
    
    plotdata(normal.T, xNormal.T, xNormalCentered, projectedNormalCentered, scaledNormal, rotNormMin.T, rotNormMax.T)
    plt.show()

print("Normal")
runExercise(normal)
print("Uniform")
runExercise(uniform)
print("Laplacian")
runExercise(laplacian)
