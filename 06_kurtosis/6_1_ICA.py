import numpy as np
import matplotlib.pyplot as plt



#defines
samplerate = 8192
N = 3
p = 18000
f_decay = 0.9995

#load the data
dataSet1 = np.loadtxt('./sounds/sound1.dat')
dataSet2 = np.loadtxt('./sounds/sound2.dat')
s = np.stack([dataSet1, dataSet2], axis=0)
print (s.shape)

#create a noise data
mean_data = np.mean(s,axis=1, keepdims=True)
std_data = np.std(s,axis=1, keepdims=True)
print (mean_data.shape)
print (mean_data)
print (std_data.shape)
print (std_data)
dataSet3 = np.random.laplace(0, 1, p);
#dataSet3 = np.random.normal(np.mean(mean_data), np.mean(std_data), p);
print (dataSet3.shape)

s = np.vstack([s, dataSet3])
print (s.shape)


# create random and invertible NxN (2x2) matrix
while True:
    A = np.random.rand(N,N)
    if np.linalg.det(A) != 0.0:
        break
print("A=" + str(A))

#mix the sources
x = np.matmul(A, s)



# remove temporal structure by permutation
x_per = x[:, np.random.permutation(range(0,p))]



#center the permuted data
mean = np.mean(x_per,axis=1, keepdims=True)
x_per_cent = x_per - mean;

#center the non-permuted data
x_cent = x - np.mean(x,axis=1, keepdims=True)


#initialize W at random
while True:
    W_init = np.random.rand(N,N)
    if np.linalg.det(W_init) != 0.0:
        break
W_nat_init = np.copy(W_init)
print("W_init=" + str(W_init))



#function that calculates f''/f'
def stepSigmoid(y):
    return 1 - 2 * (1 / (1 + np.exp(-y)))

#vectorize function
vStepSigmoid = np.vectorize(stepSigmoid)

def perform_ica(x, W, W_nat, eps=0.1):
    #perform ica
    #eps = 0.01
    eps_curr = eps
    convSpeed = np.empty([0,2])
    #for i in range(0,p):
    i=0
    nn = 0;
    while eps_curr > 0.0000001:
        #normal gradient
        dW = np.linalg.inv(W).T + np.outer(vStepSigmoid(np.dot(W,x_per_cent[:,i])),x_per_cent[:,i])
        W = W + eps * dW
        #natural gradient
        unmixed = np.dot(W_nat,x_per_cent[:,i])
        dW_nat = np.dot(np.eye(N) + np.outer(vStepSigmoid(unmixed),unmixed), W_nat)
        W_nat = W_nat + eps_curr * dW_nat
        eps_curr = eps_curr * f_decay
        i = (i+1) % p
        nn = nn +1
        if (i % 1000 == 0):
            convSpeed = np.vstack((convSpeed, [np.sum(W**2), np.sum(W_nat**2)]))

    return (W, W_nat, convSpeed, nn)

W, W_nat, convSpeed, count = perform_ica(x_per, W_init, W_nat_init)
print("\n iter=" + str(count))
print("W final=" + str(W))
print("W_nat final=" + str(W_nat))
print('ConvSpeed: ', convSpeed.shape)

# get the unmixed signals
unmixedNormal = np.matmul(W, x_cent)
unmixedNatural = np.matmul(W_nat, x_cent)



plt.figure()
plt.plot(dataSet1)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound1')
plt.figure()
plt.plot(dataSet2)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound2')
plt.figure()
plt.plot(dataSet3)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise')

plt.figure()
plt.plot(x[0,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound1 mixed')
plt.figure()
plt.plot(x[1,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound2 mixed')
plt.figure()
plt.plot(x[2,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise mixed')

plt.figure()
plt.plot(x_per[0,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound1 mixed random')
plt.figure()
plt.plot(x_per[1,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound2 mixed random')
plt.figure()
plt.plot(x_per[2,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise mixed random')

plt.figure()
plt.plot(unmixedNormal[0,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound1 unmixed normal gradient')
plt.figure()
plt.plot(unmixedNormal[1,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound2 unmixed normal gradient')
plt.figure()
plt.plot(unmixedNormal[2,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise unmixed normal gradient')

plt.figure()
plt.plot(unmixedNatural[0,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound1 unmixed natural gradient')
plt.figure()
plt.plot(unmixedNatural[1,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sound2 unmixed natural gradient')
plt.figure()
plt.plot(unmixedNatural[2,:])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Noise unmixed natural gradient')

plt.show()








