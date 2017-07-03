import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial import Voronoi, voronoi_plot_2d


## 9.1 K-means Clustering
dataSet = np.loadtxt('./cluster.dat')
dataMean = dataSet.mean(axis=1).reshape((2,1))

# Init
def init(k):
    wqInit = ((np.random.rand(2,k) - 0.5 ) * 2 + dataMean)
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

# division with 0 if no datapoint is assigned to one prototype
def choosePrototypes(mq):
#    print(np.sum(mq, axis=0))
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

# calculate error
def calculateError(mq, wq, dataSet):
    p = dataSet.shape[1]
    diff = dataSet[np.newaxis,...] - wq[np.newaxis,...].T
    nor2 = np.square(np.linalg.norm(diff, axis=1))
    err = mq*nor2.T
    err_sum = np.sum(err) / (2 * p)
#    print("sums ", err_sum)
    return err_sum

# plot error function
def plotErrorFuntion(Err, k, tmax):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(np.arange(1,tmax + 1), Err, '-')
    plt.xlabel('Iteration')
    plt.ylabel('error')
    plt.title('K = %d' % k + ' Error function')
    plt.show()

# got through different numbers of prototypes
# when prototypes are distributed in such a way that no datapoints are assigned
# to a prototype, then the algoithm fails --> division with 0
for k in range(2,9):
    wqInit, tmax = init(k)
    wq = np.copy(wqInit)
 #   print(k)
    Err = []

    for t in range(tmax):
        mq = assignDatapoints(wq, k)
        wq = choosePrototypes(mq)
        plot(k, t)
        plt.show()
        Err.append(calculateError(mq, wq, dataSet))
    
    plotErrorFuntion(Err, k, tmax)
# compute Voronoi tesselation
    if k > 3:
        try:
            vor = Voronoi(wq.T)
            voronoi_plot_2d(vor)
        except:
            print("NAN in prototypes.")
plt.show()
    

