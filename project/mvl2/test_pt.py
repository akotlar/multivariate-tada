import numpy as np
import numpy.random as rand
from mvtada_pt import MVTadaPT

rand.seed(2021)

N = 10000
p = 8
K = 2

Z_true = rand.randint(K,size=N)
Lambda = rand.randint(50,size=(K,p))
X = np.zeros((N,p))
for i in range(N):
    X[i] = rand.poisson(Lambda[int(Z_true[i])])


model = MVTadaPT(K=K)
model.fit(X.astype(np.float32),
            progress_bar=False)
            #Lamb_init=Lambda.astype(np.float32),
            #pi_init=np.zeros(K).astype(np.float32))
Z_hat = model.predict(X)


print(np.mean(X,axis=0))
print('Estimated clusters')
print(model.pi)

print('Estimated')
print(model.Lambda)
print('True')
print(Lambda)
