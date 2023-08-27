import numpy as np
import numpy.random as rand
import torch
from tqdm import trange
from torch import nn
from torch.distributions import Dirichlet
from copy import deepcopy
import scipy.stats as st
from sklearn.mixture import GaussianMixture


class MVTadaPT(object):
    def __init__(self, K=4, training_options={}):
        self.K = int(K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, data, progress_bar=True,Lamb_init=None,pi_init=None):
        """

        Parameters
        ----------

        Returns
        -------

        """
        td = self.training_options
        K = self.K

        N, p = data.shape
        X = torch.tensor(data)

        # Initialize variables
        if Lamb_init is None:
            model = GaussianMixture(K)
            model.fit(data)
            Lamb_init = model.means_.astype(np.float32)
            #Lamb_init = 10*rand.randn(K, p).astype(np.float32)
        if pi_init is None:
            pi_init = rand.randn(K).astype(np.float32)
        lambda_latent = torch.tensor(Lamb_init, requires_grad=True)
        pi_logits = torch.tensor(pi_init, requires_grad=True)

        trainable_variables = [lambda_latent, pi_logits]

        m_d = Dirichlet(torch.tensor(np.ones(self.K) * self.K))

        optimizer = torch.optim.SGD(
            trainable_variables, lr=td["learning_rate"], momentum=0.99
        )

        mse = nn.MSELoss()
        L1 = nn.L1Loss()
        smax = nn.Softmax()
        softplus = nn.Softplus()

        myrange = trange if progress_bar else range

        zeros_lamb = torch.zeros(K,p)

        self.losses_likelihoods = np.zeros(td['n_iterations'])

        k_mat = torch.tensor(-50*np.ones((td['batch_size'],p)))

        norm_x = np.mean(data)

        for i in myrange(td["n_iterations"]):
            idx = rand.choice(N, td["batch_size"], replace=False)
            X_batch = X[idx]  # Batch size x

            Lambda_un = softplus(lambda_latent)
            Lambda_n = torch.transpose(Lambda_un,0,1)/torch.mean(Lambda_un,axis=1)
            Lambda_ = norm_x*torch.transpose(Lambda_n,0,1)
            mu = .5
            pi_ = smax(pi_logits)*mu + (1-mu)/K

            loss_logits = 0.001 * mse(
                pi_logits, torch.zeros(K)
            )  # Don't allow explosions
            loss_prior_pi = -1.0 * m_d.log_prob(pi_) / N 

            # Log likelihood gamma(alpha,beta)\propto x^{alpha-1}e^{-beta x}
            term1 = 4.0 * torch.log(Lambda_)
            term2 = -5.0 * Lambda_
            loss_prior_lambda = 1 / N * torch.sum(term1 + term2)

            loglikelihood_each = [
                torch.square(X_batch-Lambda_[k]) for k in range(K)
            ]
            #loglikelihood_each = [
            #    X_batch * torch.log(Lambda_[k]) - Lambda_[k] for k in range(K)
            #]
            loglikelihood_each_trunc = [
                torch.maximum(k_mat,mat) for mat in loglikelihood_each
            ]

            loglikelihood_sum = [torch.sum(mat, axis=1) for mat in loglikelihood_each]#_trunc]
            loglikelihood_mean = [torch.mean(mat) for mat in loglikelihood_sum]
            loglike_vec = torch.stack(loglikelihood_mean)
            loglike_components = loglike_vec + torch.log(pi_)
            loss_likelihood = -1*torch.logsumexp(loglike_components,0)
            #loss_likelihood_trunc = torch.minimum(k_mat,loss_likelihood)
            #loss_likelihood = -1 * torch.sum(like_components)
            
            loss_sparse = L1(Lambda_,zeros_lamb)
            #loss_sparse = torch.mean(torch.pow(Lambda_,.5))
            ns = 1000

            loss = (
                loss_logits
                + 10*loss_prior_pi
                + loss_sparse*ns
                #+ loss_prior_lambda
                + loss_likelihood
            )
            if i % 500 == 0:
                print('_---------------')
                print(loglikelihood_each[0][0,0])
                print(i,loss_likelihood,loss_prior_pi,ns*loss_sparse)
                #print(pi_)
                #print(pi_logits)
            self.losses_likelihoods[i] = loss_likelihood.detach().numpy()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.Lambda = Lambda_.detach().numpy()
        self.pi = pi_.detach().numpy()

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
        default_options = {
            "n_iterations": 30000,
            "batch_size": 200,
            "learning_rate": 5e-5,
        }
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops
