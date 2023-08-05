import numpy as np
import numpy.random as rand
import torch
from tqdm import trange
from torch import nn
from torch.distributions import Dirichlet,


class MVTadaPT(object):

    def __init__(self, K=4, training_options={}):
        self.K = int(K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self,data):

        p = data.shape[1]
        Lamb_init = rand.randn(K,p)
        lambda_log = torch.tensor(Lamb_init,requires_grad=True)
        pi_ = torch.tensor(pi_init,requires_grad=True)

        m_d = Dirichlet(torch.tensor(np.ones(self.K)/self.K))
        m_p = Dirichlet(torch.tensor(np.ones(self.K)/self.K))

    

