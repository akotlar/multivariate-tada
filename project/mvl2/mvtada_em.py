"""
This fits a Poisson we can ea

"""
import numpy as np
import numpy.random as rand
from copy import deepcopy
import scipy.stats as st
from sklearn.mixture import GaussianMixture
from tqdm import trange

import torch
from torch import nn
from torch.nn import PoissonNLLLoss


class MVTadaPoissonEM(object):
    def __init__(self, K=4, training_options={}):
        self.K = int(K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, X, progress_bar=True, Lamb_init=None, pi_init=None):
        """
        Fits a model given 

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The count data

        progress_bar

        Lamb_init : 

        pi_init : 

        Returns
        -------
        self
        """
        td = self.training_options
        K = self.K

        N, p = X.shape

        # Initialize variables
        if Lamb_init is None:
            model = GaussianMixture(K)
            model.fit(X)
            Lambda_ = model.means_
        else:
            Lambda_ = Lamb_init

        if pi_init is None:
            pi_ = rand.dirichlet(np.ones(K))
        else:
            pi_ = pi_init

        myrange = trange if progress_bar else range

        for i in myrange(td["n_iterations"]):
            # E step compute most likely categories
            Zprobs = np.zeros((N, K))
            for j in range(K):
                log_pmf = st.poisson.logpmf(X, Lambda_[j])
                log_pmf_sum = np.sum(log_pmf, axis=1)
                Zprobs[:, j] = log_pmf_sum + np.log(pi_[j])
            Ez = np.argmax(Zprobs, axis=1)

            # M step, maximize parameters
            for j in range(K):
                pi_[j] = np.mean(Ez == j)
                Lambda_[j] = np.mean(X[Ez == j], axis=0)

            eps = 0.001
            pi_ = pi_ * (1 - K * eps) + np.ones(K) * eps

        self.Lambda = Lambda_
        self.pi = pi_
        return self

    def predict(self, data):
        """

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        log_proba = self.predict_logproba(data)
        z_hat = np.argmax(log_proba, axis=1)
        return z_hat

    def predict_logproba(self, data):
        """

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        N = data.shape[0]
        log_proba = np.zeros((N, self.K))
        for i in range(N):
            for k in range(self.K):
                log_proba[i, k] = np.sum(
                    st.poisson.logpmf(data[i], self.Lambda[k])
                )
        return log_proba

    def _fill_training_options(self, training_options):
        """
        This fills any relevant parameters for the learning algorithm

        Parameters
        ----------
        training_options : dict

        Returns
        -------
        tops : dict
        """
        default_options = {"n_iterations": 100}
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops


class MVTadaZPoissonEM(object):
    def __init__(self, K=4, training_options={}):
        self.K = int(K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, X, progress_bar=True, Lamb_init=None, pi_init=None):
        """
        Fits a model given 

        Parameters
        ----------
        X : np.array-like,shape=(N,p)
            The count data

        progress_bar

        Lamb_init : 

        pi_init : 

        Returns
        -------
        self
        """
        td = self.training_options
        K = self.K

        N, p = X.shape

        # Initialize variables
        if Lamb_init is None:
            model = GaussianMixture(K)
            model.fit(X)
            Lambda_list_l = []
            for i in range(K):
                Lambda_list_l.append(
                    torch.tensor(
                        1.2*model.means_[i].astype(np.float32)+.1, requires_grad=True
                    )
                )
        else:
            Lambda_ = torch.tensor(
                Lamb_init.astype(np.float32), requires_grad=True
            )

        myrange = trange if progress_bar else range

        pct_0 = np.mean(X==0,axis=0)
        pct_0 = .98*pct_0 + .01
        pct_0_inv = np.log(pct_0/(1-pct_0))

        alpha_list_l = [
            torch.tensor(pct_0_inv, requires_grad=True) for i in range(K)
        ]


        mse = nn.MSELoss()
        smax = nn.Softmax()
        softplus = nn.Softplus()
        sigmoid = nn.Sigmoid()

        data = torch.tensor(X.astype(np.float32))
        pi_ = 1/K*np.ones(K)
        myloss = PoissonNLLLoss(log_input=False,full=True,reduction='none')

        for i in myrange(td["n_iterations"]):
            # Lambda_ = softplus(Lambda_l)
            # alpha_ = sigmoid(alpha_l)
            # E step compute most likely categories
            Zprobs = np.zeros((N, K))
            for k in range(K):
                Lambda_k = Lambda_list_l[k]
                alpha_k = .8*sigmoid(alpha_list_l[k]) # Prob of being 0

                log_likelihood_poisson = -1*myloss(Lambda_k,data)
                log_likelihood_poisson_w = log_likelihood_poisson + torch.log(1-alpha_k)

                log_likelihood_0 = torch.log(alpha_k + (1-alpha_k)*torch.exp(-1*Lambda_k))
                log_likelihood_0_b = torch.broadcast_to(log_likelihood_0,data.shape)

                log_likelihood_point = torch.where(data!=0,log_likelihood_poisson_w,log_likelihood_0_b)

                log_marginal = torch.sum(log_likelihood_point, axis=1)

                z_probs_k = log_marginal + np.log(pi_[k])
                Zprobs[:, k] = z_probs_k.detach().numpy()

            Ez = np.argmax(Zprobs, axis=1)

            # M step, maximize parameters
            for k in range(K):
                pi_[k] = np.mean(Ez == k)
                data_sub = data[Ez == k]  # Selects relevant data

                trainable_variables = [Lambda_list_l[k], alpha_list_l[k]]

                optimizer = torch.optim.SGD(
                    trainable_variables, lr=td["learning_rate"], momentum=0.8
                )

                for j in range(td["n_inner_iterations"]):
                    Lambda_k = Lambda_list_l[k]
                    alpha_k = .8*sigmoid(alpha_list_l[k]) # Prob of being 0

                    log_likelihood_poisson = -1*myloss(Lambda_k,data_sub)
                    log_likelihood_poisson_w = log_likelihood_poisson + torch.log(1-alpha_k)

                    log_likelihood_0 = torch.log(alpha_k + (1-alpha_k)*torch.exp(-1*Lambda_k))
                    log_likelihood_0_b = torch.broadcast_to(log_likelihood_0,data_sub.shape)

                    log_likelihood_point = torch.where(data_sub!=0,log_likelihood_poisson_w,log_likelihood_0_b)

                    log_marginal = torch.sum(log_likelihood_point, axis=1)
                    loss = -1 * torch.mean(log_marginal)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    Lambda_k.requires_grad_(False)
                    Lambda_k[Lambda_k<0] = 0
                    Lambda_k.requires_grad_(True)

        self.pi = pi_
        self.Lambda = -1000*np.ones((K, p))
        self.Alpha = -1000*np.ones((K, p))
        for k in range(self.K):
            Lambda_k = Lambda_list_l[k]
            alpha_k = 0.8*sigmoid(alpha_list_l[k])

            self.Lambda[k] = Lambda_k.detach().numpy()
            self.Alpha[k] = alpha_k.detach().numpy()

        return self

    def predict(self, data):
        """

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        log_proba = self.predict_logproba(data)
        z_hat = np.argmax(log_proba, axis=1)
        return z_hat

    def predict_logproba(self, data):
        """

        Parameters
        ----------
        data : np.array-like,shape=(N_variants,n_phenotypes)
            The data of counts, variants x phenotypes

        Returns
        -------
        z_hat : np.array-like,shape=(N_samples,)
            The cluster identities
        """
        N = data.shape[0]
        log_proba = np.zeros((N, self.K))
        for i in range(N):
            for k in range(self.K):
                log_prob_poisson = st.poisson.logpmf(data[i],self.Lambda[k])
                log_prob_0 = np.log(self.Alpha[k] + (1-self.Alpha[k])*np.exp(-1*self.Lambda[k]))
                probs = log_prob_poisson.copy()
                probs[data[i]==0] = log_prob_0[data[i]==0]
                log_proba[i,k] = np.sum(probs)
        return log_proba

    def _fill_training_options(self, training_options):
        """
        This fills any relevant parameters for the learning algorithm

        Parameters
        ----------
        training_options : dict

        Returns
        -------
        tops : dict
        """
        default_options = {
            "n_iterations": 2000,
            "n_inner_iterations": 50,
            "learning_rate": 5e-4,
        }
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops
