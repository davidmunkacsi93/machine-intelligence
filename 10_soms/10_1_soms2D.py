import numpy as np
import matplotlib.pyplot as plt

## 10.1 Self-organizing maps 1D

#create random data set
dataSet = np.random.uniform(size=(2,1000))
dataSet[0,:] = dataSet[0,:] * 2.0
dataMean = dataSet.mean(axis=1).reshape((2,1))
p = dataSet.shape[1]

# Init
def init(k):
    wqInit = ((np.random.rand(2,k) - 0.5) + dataMean)
    return wqInit

# Optimization
def assignDatapoint(x, wq):
    diff = x - wq.T
    nor = np.linalg.norm(diff, axis=1)
    mqIdx = np.argmin(nor, axis=0)
    return mqIdx

def updatePrototypes(wq, x, mqIdx, eps, sig):
    diff = (x.T - wq.T).T
    nbf = calcNeighborhoodFunction(wq,np.arange(float(wq.shape[1])), mqIdx, sig)
    dw = eps * nbf * diff
    wq = wq + dw
    return wq


def calcNeighborhoodFunction(w,q,p,sig):
    diff = q - float(p)
    squ = np.square(diff)
    inp = - squ / (2.0 * sig * sig);
    return np.exp(inp)

def annealParam(eps,tau, tcurr, tmax):
    if t > tmax/4:
        eps = tau * eps
    return eps

def getError(wq):
    mq = assignDatapoints(wq, wq.shape[1])
    #print(mq.shape)
    diff = dataSet[np.newaxis,...] - wq[np.newaxis,...].T
    nor = np.square(np.linalg.norm(diff, axis=1))
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


# Plot dataset
def plot(k, t):
    plt.figure()
    plt.scatter(dataSet[0], dataSet[1])
    plt.scatter(wqInit[0], wqInit[1], color='black')
    plt.scatter(wq[0], wq[1], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('K = %d' % k + ', Iteration %d' %t)

def plot2(w, dataSet):
    m = assignDatapoints(w, w.shape[1])
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, w.shape[1])]
    plt.figure()
    for i in range(w.shape[1]):
        plt.scatter(dataSet[0,np.nonzero(m[:,i])], dataSet[1,np.nonzero(m[:,i])], linewidth='0')
        plt.scatter(w[0,i], w[1,i], edgecolor='black', linewidth='1')
    #plt.scatter(wq[0], wq[1], color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('K = %d' % w.shape[1])


def plotError(errArr):
    plt.figure()
    plt.plot(errArr)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Error over Iterations')



#Init
kArr = [4, 8, 16, 32, 64, 128]
eps = 0.2
tau = 0.9999
sig = 0.5

tmax = p
errArr = np.zeros([tmax+1])
sfdIndices = np.random.permutation(p)



#plot(k, 0)

for k in kArr:
    wqInit = init(k)
    wq = np.copy(wqInit)
    errArr[0] = getError(wq);
    
    for t in range(tmax):
        #assign datapoint to prototype
        idx = assignDatapoint(dataSet[:,sfdIndices[t % p]], wq)
        
        #update prototype locations
        wq = updatePrototypes(wq, dataSet[:,sfdIndices[t % p]], idx, eps, sig)

        #anneal epsilon and sigma
        eps = annealParam(eps, tau, t, tmax)
        sig = annealParam(sig, tau, t, tmax)

        #get the Error
        errArr[t+1] = getError(wq)


    plot2(wq, dataSet)
    plotError(errArr)

plt.show()






