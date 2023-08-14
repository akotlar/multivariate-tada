import numpy as np
import numpy.random as rand
from copy import deepcopy
import scipy.stats as st
from sklearn.mixture import GaussianMixture
from tqdm import trange


class MVTadaEM(object):
    def __init__(self, K=4, training_options={}):
        self.K = int(K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, X, progress_bar=True,Lamb_init=None,pi_init=None):
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
            #model = GaussianMixture(K)
            #model.fit(X)
            #Lambda_ = model.means_
            Lambda_ = np.abs(rand.randn(K,p))
        else:
            Lambda_ = Lamb_init

        if pi_init is None:
            pi_ = rand.dirichlet(np.ones(K))
        else:
            pi_ = pi_init

        myrange = trange if progress_bar else range

        for i in myrange(td["n_iterations"]):
            # E step compute most likely categories
            Zprobs = np.zeros((N,K))
            for j in range(K):
                log_pmf = st.poisson.logpmf(X,Lambda_[j])
                log_pmf_sum = np.sum(log_pmf,axis=1)
                Zprobs[:,j] = log_pmf_sum + np.log(pi_[j])
            Ez = np.argmax(Zprobs,axis=1)

            # M step, maximize parameters
            for j in range(K):
                pi_[j] = np.mean(Ez==j)
                Lambda_[j] = np.mean(X[Ez==j],axis=0)

            eps = .001
            pi_ = pi_*(1-K*eps) + np.ones(K)*eps

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
