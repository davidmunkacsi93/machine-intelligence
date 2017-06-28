import numpy as np
import itertools as itt
import matplotlib.pyplot as plt

#returns all permutations of -1 and 1 in a set of length n
def getPermut(n):
    return np.fromiter(itt.chain.from_iterable(itt.product([1,-1], repeat=n)),int).reshape(2**n,n)

#returns the energy for a fully connected network
def calcEnergy(s, w):
    E = -0.5 * np.dot(s.T, np.dot(w, s))
    return (E)

#returns the energy for one node of the network
def calcEnergyOfElement(s,w,i):
    E = -0.5 * s[i] * np.dot(w,s)[i]
    return E

#returns the value of the partition function
def calcPartitionFunction(n,w,beta):
    s = getPermut(n)
    sum = 0
    for i in range(s.shape[0]):
        sum = sum + np.exp( -1.0 * beta * calcEnergy(s[i,:],w))
    return (sum)

#returns the probablity that the network is in state s
def calcProbability(s,w, beta):
    z = calcPartitionFunction(s.shape[0],w, beta)
    #print("Z=" + str(z))
    p = (1.0 / z) * np.exp( -1.0 * beta * calcEnergy(s,w))
    return (p)


N = 6

#initialize the weights
W = np.random.rand(N,N)
W = (W + W.T) * 0.5
W = W - np.diag(W.diagonal())
print("Weights:" + str(W))

#initialize the state vector
S_init = np.random.choice([-1,1],6)

#initialize optimization parameters
Beta_init = 1.05
Tau = 1.05
maxIter = 50
M = np.array([1, 500])

arrBeta = np.zeros([M.shape[0], maxIter])
arrE = np.zeros([M.shape[0], maxIter])

for h in range(M.shape[0]):
    m = M[h]
    S = S_init
    Beta=Beta_init
    for i in range(maxIter):
        for j in range(m):
            #select a node randomly
            idx = np.random.choice(N,1)[0]
            #print("Selected idx: " + str(idx))

            #determine the energies fot s_i and -s_i
            E_s_i = calcEnergyOfElement(S,W,idx)
            #print("energy_s_i: " + str(E_s_i))
    
            deltaE = -2.0 * E_s_i
            #print("deltaE: " + str(deltaE))
            #print("Beta: " + str(Beta))
            tmp = (np.exp(Beta * deltaE))
            #print("tmp:" + str(tmp))
            prob = 1.0 / (1.0 + tmp)
            #print("Keep vs Flip = " + str(1.0-prob) + " vs. " + str(prob))
            newValue = np.random.choice([S[idx],-1 * S[idx]],1, p=[1.0-prob, prob])[0]
            #print("old value= " + str(S[idx]) + " new value= " + str(newValue))
    
            #assign new value
            S[idx] = newValue
            #print("S=" + str(S))
    
        #record values
        arrBeta[h,i] = 1.0 / Beta
        arrE[h,i] = calcEnergy(S, W)
        #print("arrE[i]=" + str(arrE[i]))

        #increment Beta
        Beta = Beta * Tau


for h in range(M.shape[0]):
    plt.figure()
    plt.plot(arrBeta[h,:])
    plt.xlabel('Iterations')
    plt.ylabel('Amplitude')
    plt.title('Beta for M=' + str(M[h]))
    plt.figure()
    plt.plot(arrE[h,:])
    plt.xlabel('Iterations')
    plt.ylabel('Amplitude')
    plt.title('Energy for M=' + str(M[h]))


arrStateEnergy = np.zeros(2**N)
states = getPermut(N)
for i in range(states.shape[0]):
    arrStateEnergy[i] = calcEnergy(states[i,:], W)

#plt.figure()
#plt.bar(range(2**N),arrStateEnergy)
#plt.xlabel('State')
#plt.ylabel('Energy')
#plt.title('Energy of States')

betas = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
arrStateProbabs = np.zeros([betas.shape[0],2**N])
states = getPermut(N)
for i in range(betas.shape[0]):
    for j in range(states.shape[0]):
        arrStateProbabs[i,j] = calcProbability(states[j,:], W, betas[i])
    
    #plt.figure()
    #plt.bar(range(2**N),arrStateProbabs[i,:])
    #plt.xlabel('State')
    #plt.ylabel('Probability')
    #plt.title('Probability of States for beta =' + str(betas[i]))

plt.show()
