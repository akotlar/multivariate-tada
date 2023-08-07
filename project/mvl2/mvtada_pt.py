import numpy as np
import numpy.random as rand
import torch
from tqdm import trange
from torch import nn
from torch.distributions import Dirichlet
from copy import deepcopy
import scipy.stats as st


class MVTadaPT(object):
    def __init__(self, K=4, training_options={}):
        self.K = int(K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, data, progress_bar=True):
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
        Lamb_init = rand.randn(K, p).astype(np.float32)
        pi_init = rand.randn(K).astype(np.float32)
        lambda_latent = torch.tensor(Lamb_init, requires_grad=True)
        pi_logits = torch.tensor(pi_init, requires_grad=True)

        trainable_variables = [lambda_latent, pi_logits]

        m_d = Dirichlet(torch.tensor(np.ones(self.K) / self.K))
        m_d = Dirichlet(torch.tensor(np.ones(self.K) / self.K))

        optimizer = torch.optim.SGD(
            trainable_variables, lr=td["learning_rate"], momentum=0.9
        )

        mse = nn.MSELoss()
        smax = nn.Softmax()
        softplus = nn.Softplus()

        myrange = trange if progress_bar else range

        for i in myrange(td["n_iterations"]):
            idx = rand.choice(N, td["batch_size"], replace=False)
            X_batch = X[idx]  # Batch size x

            Lambda_ = softplus(lambda_latent)
            pi_ = smax(pi_logits)

            loss_logits = 0.001 * mse(
                pi_logits, torch.zeros(K)
            )  # Don't allow explosions
            loss_prior_pi = -1.0 * m_d.log_prob(pi_) / N

            # Log likelihood gamma x^{alpha-1}e^{-beta x}
            term1 = 4.0 * torch.log(Lambda_)
            term2 = -5.0 * Lambda_
            loss_prior_lambda = 1 / N * torch.sum(term1 + term2)

            likelihood_each = [
                X_batch * torch.log(Lambda_[k]) - Lambda_[k] for k in range(K)
            ]
            [torch.sum(mat, axis=1) for mat in likelihood_each]
            likelihood_mean = [torch.mean(mat) for mat in likelihood_each]
            like_vec = torch.stack(likelihood_mean)
            like_components = like_vec + torch.log(pi_)
            loss_likelihood = -1 * torch.sum(like_components)

            loss = (
                loss_logits
                + loss_prior_pi
                + loss_prior_lambda
                + loss_likelihood
            )

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
            "n_iterations": 5000,
            "batch_size": 100,
            "learning_rate": 1e-3,
        }
        tops = deepcopy(default_options)
        tops.update(training_options)
        return tops
