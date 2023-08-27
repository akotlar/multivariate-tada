import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from mvtada_np import MVTadaEM
import numpy.linalg as la
from sklearn.mixture import GaussianMixture
np.set_printoptions(suppress=True)

def csim(vec1,vec2):
    v1 = np.squeeze(vec1)
    v2 = np.squeeze(vec2)
    num = np.dot(v1,v2)
    denom = la.norm(v1)*la.norm(v2)
    return num/denom

rand.seed(2021)

N = 200000
p = 45
K = 2

Z_true = rand.randint(K,size=N)
Lambda = rand.randint(50,size=(K,p))

#for i in range(10):
#    coord_1 = rand.choice(10)
#    coord_2 = rand.choice(2)
#    print(coord_1,coord_2)
#    Lambda[coord_2,coord_1] = 0
Lambda[0,:20] = 0
Lambda[1,20:] = 0

Lambda = Lambda*1.0
for k in range(K):
    Lambda[k] = Lambda[k]/np.mean(Lambda[k])*10
#Lambda[1] = Lambda[1]/np.mean(Lambda[1])*10
print('True')
print(Lambda)
print(csim(Lambda[1],Lambda[0]))

X = np.zeros((N,p))
for i in range(N):
    X[i] = rand.poisson(Lambda[int(Z_true[i])])

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
model = MVTadaEM(K=K)
model.fit(X)
print(model.Lambda)
Z_hat = model.predict(X)
print(np.mean(Z_hat==Z_true))

