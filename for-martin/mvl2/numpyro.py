from typing import Any, Tuple
import numpyro
from numpyro.distributions import Multinomial, Beta, Dirichlet, Beta, Categorical, MultivariateNormal, Exponential, HalfCauchy, LKJCholesky
from jax import random
from numpyro.infer import MCMC, NUTS
import numpy as np
import jax.numpy as jnp
import jax
import dill
import datetime
import os
import copy
import multiprocessing

numpyro.set_host_device_count(multiprocessing.cpu_count)
numpyro.enable_x64()


def get_pdhat(n_cases: np.array, n_ctrls: int):
    samplePDs = n_cases / (n_cases.sum() + n_ctrls)
    pdsAll = np.array([1 - samplePDs.sum(), *samplePDs])
    return pdsAll


def get_expected_K(sampleCategories: int):
    # mu_exp, var_exp = get_log_params(liabParams55cov["afMean"].numpy(), 1)
    # Expected number of components
    # For 2 case types it's
    # none, 1only, 2only, both
    # For 3 it's
    # none, 1only, 2only, 3only, 1&2, 1&3, 2&3, 123 (7)
    # for 4 it's
    # none, 1only, 2only, 3only, 4only, 1&2, 1&3, 1&4, 2&3, 2&4, 3&4, 123, 124, 134, 234, 1234
    # which is 4 + 4choose2 + nchoose3  + nchoose4
    pass


def mix_weights(beta: jnp.array):
    beta_cumprod = (1 - beta).cumprod(-1)
    return jnp.pad(beta, (0, 1), constant_values=1) * jnp.pad(beta_cumprod, (1, 0), constant_values=1)

# Covariates needed
# Sex of the individual
# parent of origin would be important


def model(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("prob_plate", k_hypotheses):
        pd_hat = get_pdhat(n_cases, n_ctrls)
        conc = numpyro.sample('conc', Exponential(pd_hat).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(conc))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

# WIP Probably not working yet


def model_mvn(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    m_sample_categories = data.shape[1]

    # If in pyro: conc = numpyro.sample('conc', Exponential(exponential_prior).to_event(1))
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("prob_plate", k_hypotheses):
        pd_hat = get_pdhat(n_cases, n_ctrls)
        # relatively uninformative prior
        # the covariance scaling
        theta = numpyro.sample("theta", HalfCauchy(
            np.ones(m_sample_categories)).to_event(1))
        L_omega = numpyro.sample("L_omega", LKJCholesky(
            m_sample_categories, np.ones(1)))
        c = jax.numpy.matmul(jax.numpy.diag(theta.sqrt()), L_omega)
        print("L_omega", L_omega.shape)
        # TODO: understand this note from Pyro code
        # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)

        # Vector of expectations
        probs = MultivariateNormal(pd_hat, scale_tril=L_omega)

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)


def infer(model_to_run, data, n_cases: np.array, n_ctrls: int, max_K: int, max_tree_depth: int,
          num_warmup=200, num_samples=1000, num_chains: int = 2, chain_method: str = 'vectorized') -> MCMC:
    mcmc = MCMC(
        NUTS(model_to_run, max_tree_depth=max_tree_depth),
        num_warmup=num_warmup, num_samples=num_samples,
        jit_model_args=True, num_chains=num_chains, chain_method=chain_method
    )
    mcmc.run(random.PRNGKey(12269), data, n_cases, n_ctrls, max_K)
    mcmc.print_summary()
    return mcmc


def get_inferred_params(mcmc: MCMC) -> Tuple[Any, Any]:
    posterior_samples = mcmc.get_samples()

    beta = posterior_samples['beta']

    weights = mix_weights(beta)
    print("probs mean", posterior_samples["probs"].mean(0))
    print("inferred stick-breaking weights mean: ", weights.mean(0))
    print("inferred stick-breaking weights stdd: ", weights.std(0))

    return posterior_samples, weights


def run(sim_data, run_params, folder_prefix: str = "") -> Tuple[MCMC, Tuple]:
    mcmc = infer(**run_params)
    inferred_params = get_inferred_params(mcmc)

    folder = datetime.datetime.now().strftime('%h-%d-%y-%H-%M-%S')
    if folder_prefix:
        folder = f"{folder_prefix}_{folder}"

    os.mkdir(folder)

    with open(f"{folder}/inferred_params.pickle", "wb") as f:
        dill.dump(inferred_params, f)

    with open(f"{folder}/mcmc.pickle", "wb") as f:
        mcmc_to_save = copy.deepcopy(mcmc)
        mcmc_to_save.sampler._sample_fn = None  # pylint: disable=protected-access
        mcmc_to_save.sampler._init_fn = None  # pylint: disable=protected-access
        mcmc_to_save.sampler._constrain_fn = None  # pylint: disable=protected-access
        mcmc_to_save._cache = {}  # pylint: disable=protected-access
        dill.dump(mcmc_to_save, f)

    with open(f"{folder}/sim_data.pickle", "wb") as f:
        dill.dump(sim_data, f)

    with open(f"{folder}/run_params.pickle", "wb") as f:
        dill.dump(run_params, f)

    return mcmc, inferred_params
