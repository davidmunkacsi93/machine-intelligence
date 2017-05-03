# (a) find 20 principal components
dataDyn = pd.read_csv("expDat.txt", index_col=0)
meanVectorDyn = np.mean(dataDyn,axis=0)
centeredDataDyn = dataDyn - meanVectorDyn

covarianceMatrixDyn = np.cov(centeredDataDyn.T)
eigenvaluesDyn, eigenvectorsDyn = np.linalg.eig(covarianceMatrixDyn)

orderedIndicesDyn = np.argsort(eigenvaluesDyn)[::-1]
orderedEigenvaluesDyn = eigenvaluesDyn[orderedIndicesDyn]
orderedEigenvectorsDyn = eigenvectorsDyn[orderedIndicesDyn]

# (b) plotting the temporal evolution
pcaDataDyn1 = np.dot(centeredDataDyn, orderedEigenvectorsDyn[0,:])
pcaDataDyn2 = np.dot(centeredDataDyn, orderedEigenvectorsDyn[1,:])

colorDyn = np.arange(100)
plt.scatter(pcaDataDyn1, pcaDataDyn2, c=colorDyn, cmap="cool")
plt.colorbar()
plt.title("Scatter plot spanned by the first two PCs")
plt.xlabel("1st PC")
plt.ylabel("2nd PC")
plt.show()

## still missing 'use that color code also in the line plots to highlight the relationship to the scatter plots'
plt.plot(pcaDataDyn1, label="1st PC", color='purple')
plt.plot(pcaDataDyn2, label="2nd PC")
plt.xlabel("time")
plt.title("Line plot of the projections onto the first PC and second PC")
plt.legend()
plt.show()

# (c) shuffling the data
newDataDyn = dataDyn.as_matrix()
for i in range(0,20):
    np.random.shuffle(newDataDyn[:,i])
    
# (d) covariance matrices and scree plots
newMeanVectorDyn = np.mean(newDataDyn,axis=0)
newCenteredDataDyn = newDataDyn - newMeanVectorDyn
newCovarianceMatrixDyn = np.cov(newCenteredDataDyn.T)
newEigenvaluesDyn, newEigenvectorsDyn = np.linalg.eig(newCovarianceMatrixDyn)
newOrderedIndicesDyn = np.argsort(newEigenvaluesDyn)[::-1]
newOrderedEigenvaluesDyn = newEigenvaluesDyn[newOrderedIndicesDyn]

plt.imshow(covarianceMatrixDyn, cmap="cool")
plt.colorbar()
plt.show()
plt.imshow(newCovarianceMatrixDyn, cmap="cool")
plt.colorbar()
plt.show()

plt.plot(orderedEigenvaluesDyn, 'o',label="original", color='purple')
plt.plot(newOrderedEigenvaluesDyn, 'o',label="shuffled") 
plt.show()

print("The scrambled data is now uncorrelated and with no underlying structure in the data anymore the screeplots show almost equally sized eigenvalues")

# (e) What would be the result if shuffling the data points in the same sequence for all columns (that is randomizing the row order)? 

print("It would have the same result, because the datapoints would be no longer correlated in a temporal way.")