import numpy as np
import math

numNodes = 6

# W from exercise 8.1
W = 100 * np.random.rand(numNodes, numNodes) - 50
W = W + W.T
np.fill_diagonal(W, 0)
#print(W)
#print(W[0,:])

# 8.2 Initialization
def init(numNodes):
    beta = 1
    tau = 1.25
    t_max = 250
    eps = 1e-4
    state = 2 * np.random.rand(numNodes) - 1
    return beta, tau, t_max, eps, state

beta, tau, t_max, eps, initstate = init(6)
#print ("state: ",state)

# 8.2 Optimization
historyBetaEnergy = np.empty((2,0))

def calculateEnergy():
    en = 0
    for i in range(numNodes):
        for j in range (numNodes):
            en += -0.5 * W[i,j] * state[i] * state[j]
    return en

def saveBetaAndEnergy():
    energy = calculateEnergy()
#    print("Energy: ", energy)
#    print(historyBetaEnergy.shape, np.asarray([beta, energy]).reshape((2,1)).shape)
    global historyBetaEnergy 
    historyBetaEnergy = np.append(historyBetaEnergy, np.asarray([beta, energy]).reshape((2,1)), axis=1)
    
def computeMeanField():
    E = np.empty((numNodes))
    for i in range(numNodes):
        e = 0
        for j in range(numNodes):
            e += -W[i,j] * state[j] 
        np.append(E, e)
    print("E: ", E)
    return E

def updateState(e):
    return np.tanh(-beta * e)


for t in range(t_max):
    state = initstate
    e_new = np.zeros((numNodes))
    e_old = np.ones((numNodes)) * eps + 0.1
    while (np.linalg.norm(e_old - e_new) > eps ):
        e_old = e_new
        e_new = computeMeanField()
        state = updateState(e_new)
        saveBetaAndEnergy()
        beta = beta * tau
        print(np.linalg.norm(e_old - e_new))
    
print(historyBetaEnergy)

#plotData()