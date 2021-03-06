import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 10.2
# a) Load and show
df = pd.read_csv("spiral.csv")
#print(df.head())
vis = plt.figure().gca(projection='3d')
vis.scatter(df['x'], df['y'], df['z'])
vis.view_init(elev=10., azim=25.)
vis.set_xlabel('x')
vis.set_ylabel('y')
vis.set_zlabel('z')
plt.show()

# b)
# adapt SOM
K = np.asarray([16, 32, 64, 128])
eps = 0.25
tau = 0.9
sig = 2

tmax = df.shape[0]
sfdIndices = np.random.permutation(tmax)
dataSet = np.asarray([df['x'], df['y'], df['z']])

# c)
# init map as line
def init_line(k):
    wq = np.zeros([3, k])
    wq[2] = np.linspace(0.0, 5.0, num=k)
    return wq

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
    if tcurr > tmax/4:
        eps = tau * eps
    return eps

# d)
# do annealing and plot final maps
def doAnnealing(k, initMap, eps, sig):
    wq = initMap
    for t in range(tmax):
        #assign datapoint to prototype
        idx = assignDatapoint(dataSet[:,sfdIndices[t]], wq)
        
        #update prototype locations
        wq = updatePrototypes(wq, dataSet[:,sfdIndices[t]], idx, eps, sig)

        #anneal epsilon and sigma
        eps = annealParam(eps, tau, t, tmax)
        sig = annealParam(sig, tau, t, tmax)
    return wq

def plotMap(finalMap, k):
    fig = plt.figure().gca(projection='3d')
    fig.scatter(finalMap[0,:], finalMap[1,:], finalMap[2,:])
    fig.view_init(elev=10., azim=25.)
    fig.set_xlabel('x')
    fig.set_ylabel('y')
    fig.set_zlabel('z')
    fig.set_title('finalMap for k = %d' % k)

    plt.show()

for k in K:
    initMap = init_line(k)
    finalMap = doAnnealing(k, initMap, eps, sig)
    plotMap(finalMap, k)

