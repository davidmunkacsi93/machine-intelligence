import numpy as np
import matplotlib.pyplot as plt

## 9.1 K-means Clustering
dataSet = np.loadtxt('./cluster.dat')
dataMean = dataSet.mean(axis=1).reshape((2,1))

# Init
def init(k):
    wqInit = (np.random.rand(2,k) - dataMean - 0.5) * 3
    tmax = 5
    return wqInit, tmax


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
    plt.xlabel('y')
    plt.title('K = %d' % k + ', Iteration %d' %t)

for k in range(2,9):
    wqInit, tmax = init(k)
    wq = np.copy(wqInit)

    for t in range(tmax):
        mq = assignDatapoints(wq, k)
        wq = choosePrototypes(mq)
        plot(k, t)
        plt.show
