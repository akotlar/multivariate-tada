from typing import Any, Tuple
import numpyro
from numpyro.distributions import Multinomial, Beta, Dirichlet, DirichletMultinomial, Gamma, Beta, Categorical, Uniform, MultivariateNormal, Normal, LogNormal, Exponential, HalfCauchy, LKJCholesky
from jax import random
from numpyro.infer import MCMC, NUTS, SA, Predictive
import numpy as np
import jax.numpy as jnp
import jax

numpyro.set_host_device_count(8)
numpyro.enable_x64()

def get_pdhat(nCases: np.array, nCtrls: int):
    samplePDs = nCases / (nCases.sum() + nCtrls)
    pdsAll = np.array([1 - samplePDs.sum(), *samplePDs])
    return pdsAll

# mu_exp, var_exp = get_log_params(liabParams55cov["afMean"].numpy(), 1)
# Expected number of components
# For 2 case types it's
# none, 1only, 2only, both
# For 3 it's
# none, 1only, 2only, 3only, 1&2, 1&3, 2&3, 123 (7)
# for 4 it's
# none, 1only, 2only, 3only, 4only, 1&2, 1&3, 1&4, 2&3, 2&4, 3&4, 123, 124, 134, 234, 1234
# which is 4 + 4choose2 + nchoose3  + nchoose4 
def mix_weights(beta: jnp.array):
    beta_cumprod = (1 - beta).cumprod(-1)
    return jnp.pad(beta, (0,1), constant_values=1) * jnp.pad(beta_cumprod, (1,0), constant_values = 1)

# Covariates needed
# Sex of the individual
# parent of origin would be important
def model(data, nCases: np.array, nCtrls: int, nHypotheses: int, alpha = .05):
    with numpyro.plate("beta_plate", nHypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / nHypotheses))

    with numpyro.plate("prob_plate", nHypotheses):
        pd_hat = get_pdhat(nCases, nCtrls)
        conc = numpyro.sample('conc', Exponential(pd_hat).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(conc))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

# WIP Probably not working yet
# def model_mvn(data, params, nHypotheses: int, alpha = .05):
#     kSampleCategories = data.shape[1]

#     # If in pyro: conc = numpyro.sample('conc', Exponential(exponential_prior).to_event(1))
#     with numpyro.plate("beta_plate", nHypotheses-1):
#         beta = numpyro.sample("beta", Beta(1, alpha / nHypotheses))

#     with numpyro.plate("prob_plate", nHypotheses):
#         theta = numpyro.sample("theta", HalfCauchy(np.ones(kSampleCategories)))
#         L_omega = numpyro.sample("L_omega", LKJCholesky(kSampleCategories, np.ones(1) ).to_event(1))
#         c = jax.numpy.matmul(jax.numpy.diag(theta.sqrt()), L_omega)
#         print("L_omega", L_omega.shape)
#         # TODO: understand this note from Pyro code
#         # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)

#         # Vector of expectations
#         probs = dist.MultivariateNormal(pdsAll, scale_tril=L_Omega)

#     with numpyro.plate("data", data.shape[0]):
#         z = numpyro.sample("z", Categorical(mix_weights(beta)))
#         return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def infer(model_to_run, data, params) -> MCMC:
    mcmc = MCMC(NUTS(model_to_run, max_tree_depth=8), num_warmup=200, num_samples=1000)
    mcmc.run(random.PRNGKey(12269), data, params, 12)
    mcmc.print_summary()
    return mcmc

def get_inferred_params(mcmc: MCMC) -> Tuple[Any, Any, Any, Any, Any]:
    posterior_samples = mcmc.get_samples()

    beta = posterior_samples['beta']

    weights = mix_weights(beta)
    print("probs mean", posterior_samples["probs"].mean(0))
    print("inferred stick-breaking weights mean: ", weights.mean(0))
    print("inferred stick-breaking weights stdd: ", weights.std(0))

    return posterior_samples, posterior_samples["probs"], posterior_samples["conc"], weights, beta

