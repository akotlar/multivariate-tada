from typing import Any, Tuple, Optional

import datetime
import os
import copy
import multiprocessing
import uuid
from typing import List

from jax import random
from jax.nn import softmax
import jax.numpy as jnp
import jax


import numpy as np
import dill

import numpyro
from numpyro.distributions import Multinomial, Normal, Beta, Dirichlet, Beta, Categorical, MultivariateNormal, Exponential, HalfCauchy, LKJCholesky, DirichletMultinomial
from numpyro.distributions.continuous import Uniform
from numpyro.infer import MCMC, NUTS, HMCECS, MixedHMC

numpyro.set_host_device_count(multiprocessing.cpu_count())

def set_platform(platform: str = "cpu") -> None:
    numpyro.set_platform(platform)

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


def model_conjugate(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    # numpyro.param('concentration', pd_hat)
    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.deterministic("probs", Uniform(pd_hat).to_event(1))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", DirichletMultinomial(concentration=probs[z]), obs=data)

def model(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(pd_hat))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        # print("probs[z]", probs[z])
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model_mvn(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    # means = numpyro.param("effect_means", get_pdhat(n_cases, n_ctrls))
    # z_scores = 
    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("effects", MultivariateNormal(0., 1.))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=softmax(probs[z])), obs=data)

# WIP Probably not working yet

# def guide(data):
#     alpha_q = numpyro.param("alpha_q", 15., constraint=constraints.positive)    
#     beta_q = numpyro.param("beta_q", lambda rng_key: random.exponential(rng_key),
#     numpyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

# >>> data = jnp.concatenate([jnp.ones(6), jnp.zeros(4)])
# >>> optimizer = numpyro.optim.Adam(step_size=0.0005)
# >>> svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
# >>> svi_result = svi.run(random.PRNGKey(0), 2000, data)
# >>> params = svi_result.params
# >>> inferred_mean = params["alpha_q"] / (params["alpha_q"] + params["beta_q"])

def model_mvn1(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
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


def infer(model_to_run, data, n_cases: np.array, n_ctrls: int, max_K: int, max_tree_depth: int, jit_model_args: bool,
          num_warmup: int, num_samples: int, num_chains: int, chain_method: str, hmcecs_blocks: Optional[int] = 0) -> MCMC:
    kernel = NUTS(model_to_run)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, jit_model_args=jit_model_args, num_chains=num_chains, chain_method=chain_method)
    mcmc.run(random.PRNGKey(12269), data, n_cases, n_ctrls, max_K)
    mcmc.print_summary()
    return mcmc


def get_inferred_params(mcmc: MCMC) -> Tuple[Any, Any]:
    posterior_samples = mcmc.get_samples()
    print(posterior_samples)
    beta = posterior_samples['beta']

    # if 'effects' in posterior_samples:
    #     probs = softmax(posterior_samples["probs"])

    weights = mix_weights(beta)
    # print("probs mean", posterior_samples["probs"].mean(0))
    print("inferred stick-breaking weights mean: ", weights.mean(0))
    print("inferred stick-breaking weights stdd: ", weights.std(0))

    return posterior_samples, weights


def run(sim_data, run_params, folder_prefix: str = "") -> Tuple[MCMC, Tuple]:
    mcmc = infer(**run_params)
    inferred_params = get_inferred_params(mcmc)
    suffix = uuid.uuid4()
    folder = datetime.datetime.now().strftime('%h-%d-%y-%H-%M-%S') + f"_{suffix}"
    if folder_prefix:
        folder = f"simulations/{folder_prefix}_{folder}"

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

def read_shit(path) -> List[float]:
    prevalences = None
    headers = None
    params = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('Args are: '):
                headers = line.strip().split('Args are: ')[1].split(',')
            elif line.startswith('Args val: '):
                for val in line.strip().split('Args val: ')[1].split(','):
                    try:
                        params.append(float(val))
                    except:
                        params.append(val)
            elif line.startswith('Final Observed Prevalences for this study are (Disorder1,Disorder2,Both) = '):
                prevalences = list(map(float, line.split('Final Observed Prevalences for this study are (Disorder1,Disorder2,Both) = ')[1].strip().split(',')))

    return prevalences, dict(zip(headers, params))

def select_components(weights: np.array, threshold: float = .01):
    accepted_indices = []
    accepted_weight_means = []
    accepted_weight_stds = []
    weight_means = weights.mean(0)
    weight_stds = weights.std(0)

    for i, weight in enumerate(weight_means):
        if weight >= .01:
            accepted_indices.append(i)
            accepted_weight_means.append(weight)
            # for some reason else will come out device array
            accepted_weight_stds.append(float(weight_stds[i]))

    return {"mean": accepted_weight_means, "std": accepted_weight_stds, "indices": accepted_indices}

def get_run_params_data(folder: str) -> Tuple[dict, dict]:
    run_params = None
    with open(os.path.join(folder, "run_params.pickle"), 'rb') as file:
        run_params = dill.load(file)

    sim_data = None
    with open(os.path.join(folder, "sim_data.pickle"), 'rb') as file:
        sim_data = dill.load(file)
        
    return run_params, sim_data

