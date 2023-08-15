"""


"""
import numpy as np
import numpy.random as rand
from copy import deepcopy
import scipy.stats as st
from sklearn.mixture import GaussianMixture
from tqdm import trange

import torch
from torch import nn


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
                        model.means_[i].astype(np.float32), requires_grad=True
                    )
                )
        else:
            Lambda_ = torch.tensor(
                Lamb_init.astype(np.float32), requires_grad=True
            )

        myrange = trange if progress_bar else range

        alpha_list_l = [
            torch.tensor(-10 * np.ones(p), requires_grad=True) for i in range(K)
        ]

        mse = nn.MSELoss()
        smax = nn.Softmax()
        softplus = nn.Softplus()
        sigmoid = nn.Sigmoid()

        data = torch.tensor(X.astype(np.float32))
        pi_ = 1/K*np.ones(K)

        for i in myrange(td["n_iterations"]):
            # Lambda_ = softplus(Lambda_l)
            # alpha_ = sigmoid(alpha_l)
            # E step compute most likely categories
            Zprobs = np.zeros((N, K))
            for k in range(K):
                Lambda_k = softplus(Lambda_list_l[k])
                alpha_k = sigmoid(alpha_list_l[k])

                # Poisson_component
                term_3 = -0.5 * torch.log(2 * np.pi * data)
                term_1 = -Lambda_k
                term_2 = data * torch.log(Lambda_k)
                log_likelihood_poisson = term_1 + term_2 + term_3
                log_likelihood_poisson_w = log_likelihood_poisson - torch.log(
                    alpha_k
                )
                log_marginal_p = torch.sum(log_likelihood_poisson_w, axis=1)
                log_0 = (1 - alpha_k) * torch.where(data == 0,1,0)
                log_marginal_0 = torch.sum(log_0, axis=1)
                log_marginal = log_marginal_0 + log_marginal_p
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
                    Lambda_k = softplus(Lambda_list_l[k])
                    alpha_k = sigmoid(alpha_list_l[k])

                    # Poisson_component
                    term_3 = -0.5 * torch.log(2 * np.pi * data_sub)
                    term_1 = -Lambda_k
                    term_2 = data_sub * torch.log(Lambda_k)
                    log_likelihood_poisson = term_1 + term_2 + term_3
                    log_likelihood_poisson_w = (
                        log_likelihood_poisson - torch.log(alpha_k)
                    )
                    log_marginal_p = torch.sum(log_likelihood_poisson_w, axis=1)
                    log_0 = (1 - alpha_k) * torch.where(data_sub == 0,1,0)
                    log_marginal_0 = torch.sum(log_0, axis=1)
                    log_marginal = log_marginal_0 + log_marginal_p
                    loss = -1 * torch.mean(log_marginal)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        self.pi = pi_
        self.Lambda = np.zeros((K, p))
        self.Alpha = np.zeros((K, p))
        for k in range(self.K):
            Lambda_k = softplus(Lambda_list_l[k])
            alpha_k = sigmoid(alpha_list_l[k])

            self.Lambda[k] = Lambda_k.detach().numpy()
            self.Alpha[k] = Lambda_k.detach().numpy()

            return self

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
            "n_iterations": 100,
            "n_inner_iterations": 50,
            "learning_rate": 1e-3,
        }
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops
