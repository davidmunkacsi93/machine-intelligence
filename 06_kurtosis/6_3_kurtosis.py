import numpy as np
import scipy.io as io
from sklearn.decomposition import PCA

#load data from mat file
distrib = io.loadmat("distrib.mat")

normal = distrib.get("normal")
uniform = distrib.get("uniform")
laplacian = distrib.get("laplacian")

# a) apply mixing
A = np.asarray([[4,3],[2,1]])
xNormal = np.dot(A,normal)
xUniform = np.dot(A,uniform)
xLaplacian = np.dot(A,laplacian)

# b) center to zero mean 
xNormalCentered = xNormal - xNormal.mean()
xUniformCentered = xUniform - xUniform.mean()
xLaplacianCentered = xLaplacian - xLaplacian.mean()

# c) apply PCA and project data onto PC
pca = PCA()
xNormalCenteredProjected = pca.fit_transform(xNormalCentered)
normalVar = pca.explained_variance_

pca = PCA()
xUniformCenteredProjected = pca.fit_transform(xUniformCentered)
uniformVar = pca.explained_variance_

pca = PCA()
xLaplacianCenteredProjected = pca.fit_transform(xLaplacianCentered)
laplacianVar = pca.explained_variance_

print(xNormalCenteredProjected)
# d) scale data to unit variance
scaledNormal = xNormalCenteredProjected / 1
print(xNormalCentered.shape)


##defines
#samplerate = 8192
#N = 2
#p = 18000
#
##load the data
#dataSet1 = np.loadtxt('./sounds/sound1.dat')
#dataSet2 = np.loadtxt('./sounds/sound2.dat')
#s = np.stack([dataSet1, dataSet2], axis=0)
#
#print (s.shape)
#
## create random and invertible NxN (2x2) matrix
#while True:
#    A = np.random.rand(2,2)
#    if np.linalg.det(A) != 0.0:
#        break
#print("A=" + str(A))
#
##mix the sources
#x = np.matmul(A, s)
#print(x[:,:10])
#print(x.shape)
#
## remove temporal structure by permutation 
#x_per = x[:, np.random.permutation(range(0,18000))]
#print(x_per[:,:10])
#print("x_per=" + str(x_per.shape))
#
##calculate correlation between sources and mixtures
#s_std = np.std(s,axis=1)
#x_per_std = np.std(x_per,axis=1)
#print(s_std)
#print(x_per_std) 
#
#corr = np.zeros((2, 2))
#for i in range(0,2):
#	for j in range(0,2):
#		#print("i=" + str(i) + " j=" + str(j))
#		corr[i,j] = np.cov(s[i,:], x_per[j,:])[0][1] / (s_std[i] * x_per_std[j])
#		#print(np.cov(s[i,:], x_per[j,:]))
#		#print("corr[i,j]=" + str(corr[i,j]))
#print("correlations=" + str(corr))
#
##center the data
#x_per = x_per - np.mean(x_per,axis=0);
#
##initialize W at random
#W = np.random.rand(2,2)
#W_nat = np.copy(W)
#print("W_nat=" + str(W_nat))
#def stepSigmoid(y):
#	return 1 - 2 * (1 / (1 + np.exp(-y)))
#	
#vStepSigmoid = np.vectorize(stepSigmoid)
#eps = 0.01
#for i in range(0,1):
#	W_inv = np.linalg.inv(W) 
##	print("W_inv=" + str(W_inv)) 
##	currX = x_per[:,i]
##	print("currX=" + str(currX)) 
##	inpSig = np.dot(W,x_per[:,i])
##	print("inpSig=" + str(inpSig)) 
##	print("inpSigShape=" + str(inpSig.shape)) 
##	vStepSigmoidRes = vStepSigmoid(np.dot(W,x_per[:,i]))
##	print("vStepSigmoidRes=" + str(vStepSigmoidRes)) 
##	print("vStepSigmoidResShape=" + str(vStepSigmoidRes.shape)) 
##	outer = np.outer(vStepSigmoid(np.dot(W,x_per[:,i])),x_per[:,i])
##	print("outer=" + str(outer)) 
##	print("outerShape=" + str(outer.shape)) 
#	dW = W_inv.T + np.outer(vStepSigmoid(np.dot(W,x_per[:,i])),x_per[:,i])
##	print("at iter" + str(i) + ", dW=" + str(dW))
#	W = W + eps * dW
##	print("at iter" + str(i) + ", W=" + str(W))
#	#W_nat_inv = np.linalg.inv(W_nat) 
#	#dW_nat = W_nat_inv.T + np.outer(vStepSigmoid(np.dot(W_nat,x_per[:,i])),x_per[:,i])
#	#W_nat = W_nat + eps * dW_nat * (np.matmul(W_nat.T, W_nat))
#
#print("final=" + str(A - np.linalg.inv(W)))
##print("final nat=" + str(A - np.linalg.inv(W_nat)))
#
#
#
#
#
#
#
#
