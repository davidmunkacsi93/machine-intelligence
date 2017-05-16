import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# Load and show outliners
dataWhite = pd.read_csv("pca4.csv", sep=",").as_matrix()
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.scatter(dataWhite[:,0],dataWhite[:,1])
plt.scatter(dataWhite[:,2],dataWhite[:,3], color='purple')

# Remove outliners
for i in range(495):
    if (abs(dataWhite[:,2][i]) > 10 or abs(dataWhite[:,3][i]) > 10 ):
        dataWhite = np.delete(dataWhite, (i), axis=0)

# reasonable Subset
plt.subplot(1,2,2)
plt.scatter(dataWhite[:,0],dataWhite[:,1])
plt.scatter(dataWhite[:,2],dataWhite[:,3], color='purple')
plt.show()

# PCA
meanVectorWhite = np.mean(dataWhite,axis=0)
centeredDataWhite = dataWhite - meanVectorWhite
covarianceMatrixWhite = np.cov(centeredDataWhite.T)
eigenvaluesWhite, eigenvectorsWhite = np.linalg.eig(covarianceMatrixWhite)
orderedIndicesWhite = np.argsort(eigenvaluesWhite)[::-1]
orderedEigenvaluesWhite = eigenvaluesWhite[orderedIndicesWhite]
orderedEigenvectorsWhite = eigenvectorsWhite[orderedIndicesWhite]
pcaDataWhite = np.dot(centeredDataWhite, orderedEigenvectorsWhite)
np.cov(pcaDataWhite)

# Screeplot
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.title("Screeplot")
plt.plot(orderedEigenvaluesWhite, 'o',label="original", color='purple')

# Plotting against two PC.
plt.subplot(1,2,2)
plt.scatter(pcaDataWhite[:,0], pcaDataWhite[:,1])
plt.title("Scatter plot of first two PCs")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

#Whiten
X = centeredDataWhite
E = eigenvectorsWhite
L = la.sqrtm(np.linalg.inv(np.diag(eigenvaluesWhite)))
Z = np.dot(np.dot(X,E),L)

#Heat plots
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(covarianceMatrixWhite, cmap="cool")
plt.subplot(1,3,2)
plt.imshow(np.cov(pcaDataWhite.T), cmap="cool")
plt.subplot(1,3,3)
plt.imshow(Z, cmap="cool", aspect='auto')
plt.colorbar()
plt.show()