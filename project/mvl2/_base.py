"""

"""
import numpy as np
from typing import Literal
import jax.numpy as jnp
import numpyro
from numpyro.distributions import Distribution,Poisson
import numpy.typing as npt
from numpyro.distributions import MultinomialProbs,constraints
from jax.scipy.special import gammaln

ArrayLike = npt.ArrayLike


def set_platform(platform: Literal["cpu", "gpu", "tpu"] = "cpu") -> None:
    numpyro.set_platform(platform)


def get_pdhat(n_cases: np.array, n_ctrls: int):
    samplePDs = n_cases / (n_cases.sum() + n_ctrls)
    pdsAll = np.array([1 - samplePDs.sum(), *samplePDs])
    return pdsAll


def get_weights_from_mcmc_samples_beta(beta: jnp.array):
    weights = []
    for b in beta:
        weights.append(mix_weights(b))

    return np.array(weights)


def mix_weights(beta: jnp.array):
    beta_cumprod = (1 - beta).cumprod(-1)
    return jnp.pad(beta, (0, 1), constant_values=1) * jnp.pad(
        beta_cumprod, (1, 0), constant_values=1
    )


def method_moments_estimator_gamma_shape_rate(
    empirical_prevalence_estimate: ArrayLike = None, data: ArrayLike = None
):
    assert data is not None or empirical_prevalence_estimate is not None

    if empirical_prevalence_estimate is None:
        empirical_prevalence_estimate = data.mean(0)

    try:
        sd = data.std(0)
    except:
        sd = np.std(data.numpy())

    moment_methods_shape = empirical_prevalence_estimate ** 2 / sd ** 2
    moment_methods_rate = empirical_prevalence_estimate / sd

    return moment_methods_shape, moment_methods_rate

class ProductPoisson(Distribution):
    
    def __init__(self,rates,*args,**kwargs):
        self.rates = rates
        super().__init__(*args, **kwargs)

    def log_prob(self,value):
        term1 = jnp.sum(-1*self.rates)
        term2 = jnp.sum(value*jnp.log(self.rates))
        term3 = -1*jnp.sum(gammaln(value+1))
        return term1 + term2 + term3

class ProductZIPoisson(Distribution):
    
    def __init__(self,rates,gates,*args,**kwargs):
        self.rates = rates
        self.gates = gates
        super().__init__(*args, **kwargs)

    def log_prob(self,value):
        term1 = -1*self.rates
        term2 = value*jnp.log(self.rates)
        term3 = -1*gammaln(value+1)
        log_prob_nonzero = jnp.log1p(-self.gates) + term1 + term2 + term3
        log_prob = jnp.where(value==0,jnp.log(self.gates + jnp.exp(log_prob_nonzero)),log_prob_nonzero)
        lp = jnp.sum(log_prob)
        return lp


class ZeroInflatedMultinomial(MultinomialProbs):
    """


    """

    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, *args, gate_n: int, gate_pop_af: float, **kwargs):
        super().__init__(*args, **kwargs)
        prob_zero = jax.scipy.stats.poisson.pmf(0, gate_n * gate_pop_af)

        self.gate = jnp.expand_dims(prob_zero, 1)

    def log_prob(self, value):
        ll = super().log_prob(value)
        ll = jnp.where(
            value.T == 0,
            jnp.log(self.gate + jnp.exp(ll)),
            jnp.log1p(-self.gate) + ll,
        )

        return ll

