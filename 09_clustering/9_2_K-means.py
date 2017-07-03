import numpy as np
import matplotlib.pyplot as plt

## 9.1 K-means Clustering
dataSet = np.loadtxt('./cluster.dat')
dataMean = dataSet.mean(axis=1).reshape((2,1))
p = dataSet.shape[1]

# Init
def initOnline(k):
    wqInit = ((np.random.rand(2,k) - 0.5 ) * 2 + dataMean)
    tmax = p
    return wqInit, tmax

# Optimization
def assignDatapoint(x, wq):
    diff = x - wq.T
    nor = np.linalg.norm(diff, axis=1)
    mqIdx = np.argmin(nor, axis=0)
    return mqIdx

def updatePrototype(wq, x, mqIdx, eps):
    diff = x - wq[:,mqIdx]
    dw = eps * diff
    wq[:,mqIdx] = wq[:,mqIdx] + dw
    return wq

def updateEpsilon(eps,tau, tcurr, tmax):
    if t > tmax/4:
        eps = tau * eps
    return eps

def getError(wq):
    mq = assignDatapoints(wq, wq.shape[1])
    diff = dataSet[np.newaxis,...] - wq[np.newaxis,...].T
    nor = np.linalg.norm(diff, axis=1)
    tmp = nor[np.nonzero(mq.T)]
    err = (1.0 / (2.0 * p)) * tmp.sum()
    return err



# Optimization
def assignDatapoints(wq, k):
    diff = dataSet[np.newaxis,...] - wq[np.newaxis,...].T
    nor = np.linalg.norm(diff, axis=1)
    mqIdx = np.argmin(nor, axis=0)
    mq = np.zeros((dataSet.shape[1], k))
    rows = list(range(dataSet.shape[1]))
    mq[rows, mqIdx] = 1
    return mq


def choosePrototypes(mq):
    print(np.sum(mq, axis=0))
    wqNew = np.dot(dataSet, mq) / np.sum(mq, axis=0)
    return wqNew

# Plot dataset
def plot(k, t):
    plt.figure()
    plt.scatter(dataSet[0], dataSet[1])
    plt.scatter(wqInit[0], wqInit[1], color='black')
    plt.scatter(wq[0], wq[1], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('K = %d' % k + ', Iteration %d' %t)

def plotError(errArr):
    plt.figure()
    plt.plot(errArr)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Iterations')


#for k in range(2,9):
#    wqInit, tmax = init(k)
#    wq = np.copy(wqInit)

#    for t in range(tmax):
#        mq = assignDatapoints(wq, k)
#        wq = choosePrototypes(mq)
#        plot(k, t)
#       plt.show


k = 4
eps = 0.1
tau = 0.99

wqInit, tmax = initOnline(k)
wq = np.copy(wqInit)



itr = 1
step = (p-1)/5.0

errArr = np.zeros([tmax+1])
errArr[0] = getError(wq);

sfdIndices = np.random.permutation(p)

plot(k, 0)

for t in range(tmax):
    idx = assignDatapoint(dataSet[:,sfdIndices[t]], wq)
    wq = updatePrototype(wq, dataSet[:,sfdIndices[t]], idx, eps)
    eps = updateEpsilon(eps,tau, t, tmax)

    #get the Error
    errArr[t+1] = getError(wq)

    #capture the prototypes at certain iterations
    if (t == np.rint(itr*step)):
        plot(k, t)
        itr = itr+1




#plot(k, tmax)
#plt.show()
plotError(errArr)
plt.show()



