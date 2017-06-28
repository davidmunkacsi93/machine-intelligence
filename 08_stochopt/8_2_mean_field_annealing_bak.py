import numpy as np
import matplotlib.pyplot as plt

numNodes = 6

#initialize the weights
W = np.random.rand(numNodes,numNodes)
W = (W + W.T) * 0.5
W = W - np.diag(W.diagonal())

# 8.2 Initialization
def init(numNodes):
    beta = 0.5
    tau = 1.15
    t_max = 20
    eps = 1e-10
    state = 2 * np.random.rand(numNodes) - 1
    meanField = np.zeros([6])
    return beta, tau, t_max, eps, state, meanField

beta, tau, t_max, eps, initstate, meanField = init(numNodes)

# 8.2 Optimization
historyBetaEnergy = np.empty((2,0))

def calcEnergy(state):
    egy = -0.5 * np.dot(state.T, np.dot(W, state))
    return (egy)

def saveBetaAndEnergy(state):
    energy = calcEnergy(state)
    global historyBetaEnergy 
    historyBetaEnergy = np.append(historyBetaEnergy, np.asarray([beta, energy]).reshape((2,1)), axis=1)
    
def computeMeanField(state):
    for i in range(numNodes):
        e = 0
        for j in range(numNodes):
            if j != i:
                e += -W[i,j] * state[j] 
        meanField[i] = e
        state[i] = np.tanh(-beta * e)            
    ret = meanField
    return ret, state

def updateState(e):
    st = np.tanh(-beta * e)
    return st


for t in range(t_max):
    state = np.copy(initstate)
    e_new = np.zeros((numNodes))
    e_old = np.ones((numNodes)) * eps + 0.1
    counter = 0
    nor = np.linalg.norm(e_old - e_new)
    while (nor > eps ):
        e_old = np.copy(e_new)
        e_new, state = computeMeanField(state)
        saveBetaAndEnergy(state)
        counter += 1
        nor = np.linalg.norm(e_old - e_new)
        print("Norm: ", nor)
    beta = beta * tau
    print("Done! ", counter)
    print("")
    
print(historyBetaEnergy)
plt.figure()
plt.plot(1/historyBetaEnergy[0,:])
plt.xlabel('Iterations')
plt.ylabel('Temperature')

plt.figure()
plt.plot(historyBetaEnergy[1,:])
plt.xlabel('Iterations')
plt.ylabel('Energy')

plt.show()

