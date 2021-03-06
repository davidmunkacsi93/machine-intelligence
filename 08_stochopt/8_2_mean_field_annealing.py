import numpy as np

numNodes = 6

#initialize the weights
W = np.random.rand(numNodes,numNodes)
W = (W + W.T) * 0.5
W = W - np.diag(W.diagonal())
#print(W)
#print(W[0,:])

# 8.2 Initialization
def init(numNodes):
    beta = 0.25
    tau = 1.05
    t_max = 20
    eps = 1e-9
    state = 2 * np.random.rand(numNodes) - 1
    meanField = np.zeros([6])
    return beta, tau, t_max, eps, state, meanField

beta, tau, t_max, eps, initstate, meanField = init(numNodes)
print ("state: ",initstate)
print("eps: ", eps)

# 8.2 Optimization
historyBetaEnergy = np.empty((2,0))

def calcEnergy(state):
    egy = -0.5 * np.dot(state.T, np.dot(W, state))
    return (egy)

def saveBetaAndEnergy(state):
    energy = calcEnergy(state)
#    print("Energy: ", energy)
#    print(historyBetaEnergy.shape, np.asarray([beta, energy]).reshape((2,1)).shape)
    global historyBetaEnergy 
    historyBetaEnergy = np.append(historyBetaEnergy, np.asarray([beta, energy]).reshape((2,1)), axis=1)
    
def computeMeanField(s, i):
     e = 0
     for j in range(numNodes):
         if j != i:
             e += -W[i,j] * state[j] 
             #print("e", e)
    print("e", e)
    meanField[i] = e
    #state[i] = np.tanh(np.deg2rad(-beta * e))
    state[i] = np.tanh(-beta * e)
            
    print("meanField: ", meanField)
    ret = meanField
    return ret, state

def updateState(e):
    st = np.tanh(np.deg2rad(-beta * e))
    #st = np.tanh(-beta * e)
    print("st: ", st, beta, e)
    return st


for t in range(t_max):
    state = initstate
    e_new = np.zeros((numNodes))
    e_old = np.ones((numNodes)) * eps + 0.1
    counter = 0
    nor = np.linalg.norm(e_old - e_new)
    while (nor > eps ):
        e_old = e_new
        e_new, state = computeMeanField(state)
        print("e_old  / e_new")
        print(e_old)
        print(e_new)
        #state = updateState(e_new)
        print("State: ", state)
        saveBetaAndEnergy(state)
        counter += 1
        nor = np.linalg.norm(e_old - e_new)
        print("Norm: ", nor)
    beta = beta * tau
    print("beta: ", beta)
    print("Done! ", counter)
    print("")
    
print(historyBetaEnergy)

#plotData()