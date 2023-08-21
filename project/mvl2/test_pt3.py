import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from mvtada_pt2 import MVTadaPT
import numpy.linalg as la
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
#Lambda[0,:20] = 0
#Lambda[1,20:] = 0

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


model = MVTadaPT(K=K)
model.fit(X.astype(np.float32),
            progress_bar=False)
            #Lamb_init=Lambda.astype(np.float32),
            #pi_init=np.zeros(K).astype(np.float32))
Z_hat = model.predict(X)


print('Data averages')
print(np.mean(X,axis=0))
print('Estimated clusters')
print(model.pi)
print('Model',np.mean(model.Lambda,axis=1))
print('True',np.mean(Lambda,axis=1))

print('Estimated')
print(model.Lambda)
print('True')
print(Lambda)
print('------------------------------------')
print('------------------------------------')
print(csim(model.Lambda[0],Lambda[0]))
print(csim(model.Lambda[0],Lambda[1]))
print(csim(model.Lambda[1],Lambda[0]))
print(csim(model.Lambda[1],Lambda[1]))
print('------------------------------------')
print('------------------------------------')
print(csim(model.Lambda[1],model.Lambda[0]))

plt.plot(model.losses_likelihoods)
plt.show()
