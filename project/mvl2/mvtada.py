"""

Objects
-------

Methods
-------

"""
import numpy as np
import jax.numpy as jnp
from typing import Union
import multiprocessing
from copy import deepcopy

import numpy.typing as npt
import numpyro
from numpyro.distributions import (
    Beta,
    Categorical,
    Gamma,
    Poisson,
    ZeroInflatedPoisson,
)
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.ops.indexing import Vindex

from jax import random

from _base import (
    mix_weights,
    method_moments_estimator_gamma_shape_rate,
    ProductPoisson,
)

ArrayLike = npt.ArrayLike


class MultiVariateTada(object):
    def __init__(self, max_K=4, training_options={}):
        self.max_K = int(max_K)
        self.training_options = self._fill_training_options(training_options)

    def fit(self, data, n_cases=None, n_controls=None, seed=2021):
        """
        Parameters
        ----------

        Returns
        -------
        self
        """
        random_key = random.PRNGKey(seed)

        td = self.training_options

        extra_fields = ("potential_energy", "accept_prob", "mean_accept_prob")

        numpyro.set_host_device_count(multiprocessing.cpu_count())

        model = modelPoisson

        kernel = NUTS(model,
        #kernel = DiscreteHMCGibbs(model)
            target_accept_prob=td["target_accept_prob"],
            max_tree_depth=td["max_tree_depth"],
        )
        mcmc = MCMC(
            kernel,
            num_warmup=td["num_warmup"],
            num_samples=td["num_samples"],
            jit_model_args=td["jit_model_args"],
            num_chains=1,
            thinning=td["thinning"],
        )
        model_args = {
            "data": data,
            "k_hypotheses": 5,
            "alpha": 0.05,
            "shared_dirichlet_prior": False,
        }
        mcmc.run(random_key, extra_fields=extra_fields, **model_args)

        self.samples = mcmc.get_samples()
        return self

    def _fill_training_options(self, training_options):
        """
        Parameters
        ----------
        
        Returns
        -------

        """
        default_dict = {
            "num_samples": 8000,
            "num_warmup": 2000,
            "max_tree_depth": 8,
            "thinning": 1,
            "jit_model_args": False,
            "target_accept_prob": 0.5,
        }
        z = deepcopy(training_options)
        z.update(default_dict)
        return z


def modelPoisson(
    data: jnp.array = None,
    k_hypotheses: int = 5,
    alpha: float = 0.05,
    shared_dirichlet_prior: bool = False,
    gamma_shape: Union[float, ArrayLike] = None,
    gamma_rate: Union[float, ArrayLike] = None,
):

    """

        Parameters

        ----------

        shared_dirichlet_prior: bool

            Whether each component should share the same Gamma prior.

            This results in fewer parameters to estimate, and may perform

            better when the genetic architecture allowfrom numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
from numpyro.ops.indexing import Vindex
s for it.

    """

    with numpyro.plate("beta_plate", k_hypotheses - 1):
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("lambda_plate", k_hypotheses,dim=-2):
        with numpyro.plate("lambda_plate2", 4,dim=-1):
        #lmbda = numpyro.sample("lambda", Gamma(5.0*np.ones((4,1)), 5.0*np.ones((4,1))))
            lmbda = numpyro.sample("lambda", Gamma(5.0, 5.0))

    print('A',lmbda.shape)

    pz = numpyro.deterministic("pz", mix_weights(beta))

    with numpyro.plate("individuals", data.shape[0],dim=-2):
        z = numpyro.sample("z", Categorical(pz))
        with numpyro.plate('observations',data.shape[1],dim=-1):
        #with numpyro.plate("data", data.shape[1]):
        #lmbda_z = Vindex(lmbda)[:,z]
        #print('>>>>>>>>>>>>>>>')
        #print('>>>>>>>>>>>>>>>')
        #print(data.shape)
        #print(lmbda_z.shape)
        #print(lmbda.shape)
        #print(lmbda[:,z].shape)
        #print(Vindex(lmbda)[z,:].shape)
        #print(Vindex(lmbda)[:,z].shape)
        #print('>>>>>>>>>>>>>>>')
        #print('>>>>>>>>>>>>>>>')
        #return numpyro.sample("obs",ProductPoisson(lmbda[:z]),obs=data)
        #return numpyro.sample("obs", 
        #                        Poisson(jnp.transpose(lmbda[:,z])), obs=data)
        #return numpyro.sample("obs", 
        #                        Poisson(lmbda_z), obs=data)
        #return numpyro.sample("obs", 
        #                        Poisson(jnp.squeeze(Vindex(lmbda)[z,:])), obs=data)
            return numpyro.sample("obs", 
                                Poisson(Vindex(lmbda)[z]), obs=data)

