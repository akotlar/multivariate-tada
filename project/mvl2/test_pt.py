import numpy as np
import numpy.random as rand
from mvtada_pt import MVTadaPT

N = 10000
p = 10
K = 4

Z_true = rand.randint(K,size=N)
Lambda = rand.randint(50,size=(K,p))
X = np.zeros((N,p))
for i in range(N):
    X[i] = rand.poisson(Lambda[int(Z_true[i])])


model = MVTadaPT()
model.fit(X.astype(np.float32))
Z_hat = model.predict(X)
