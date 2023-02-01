from typing import Any, Tuple, Optional

import datetime
import os
import copy
import multiprocessing
import uuid

import torch
import numpy as np
import cloudpickle

import pyro
from pyro.distributions import Exponential
from pyro.distributions.torch import Multinomial, Beta, Dirichlet, Beta, Categorical, MultivariateNormal, Uniform#, Exponential #, HalfCauchy, LKJCholesky
from pyro.infer import TraceEnum_ELBO, MCMC, NUTS, TraceGraph_ELBO, config_enumerate#, #HMCECS, MixedHMC
from torch import tensor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoDelta, AutoNormal

assert pyro.__version__.startswith('1.7.0')
pyro.set_rng_seed(0)

def set_platform(platform: str = "cpu") -> None:
    pyro.set_platform(platform)

def get_pdhat(n_cases: tensor, n_ctrls: int):
    samplePDs = n_cases / (n_cases.sum() + n_ctrls)
    return tensor([1 - samplePDs.sum(), *samplePDs])


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


def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

# Covariates needed
# Sex of the individual
# parent of origin would be important
def model(data, n_cases: tensor, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with pyro.plate("beta_plate", k_hypotheses-1):
        beta = pyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with pyro.plate("prob_plate", k_hypotheses):
        probs = pyro.sample("probs", Dirichlet(pd_hat))

    with pyro.plate("data", data.shape[0]):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        return pyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

# def model(data):
#     with pyro.plate("beta_plate", T-1):
#         beta = pyro.sample("beta", Beta(1, alpha))

#     with pyro.plate("mu_plate", T):
#         mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))

#     with pyro.plate("data", N):
#         z = pyro.sample("z", Categorical(mix_weights(beta)))
#         pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)

# def guide(data):
#     kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
#     tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)).sample([T]))
#     phi = pyro.param('phi', lambda: Dirichlet(1/10).sample([N]), constraint=constraints.simplex)

#     with pyro.plate("beta_plate", T-1):
#         q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

#     with pyro.plate("mu_plate", T):
#         q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

#     with pyro.plate("data", N):
#         z = pyro.sample("z", Categorical(phi))

# WIP Probably not working yet

# def model_mvn(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
#     m_sample_categories = data.shape[1]

#     # If in pyro: conc = numpyro.sample('conc', Exponential(exponential_prior).to_event(1))
#     with pyro.plate("beta_plate", k_hypotheses-1):
#         beta = pyro.sample("beta", Beta(1, alpha / k_hypotheses))

#     with pyro.plate("prob_plate", k_hypotheses):
#         pd_hat = get_pdhat(n_cases, n_ctrls)
#         # relatively uninformative prior
#         # the covariance scaling
#         theta = pyro.sample("theta", HalfCauchy(
#             np.ones(m_sample_categories)).to_event(1))
#         L_omega = pyro.sample("L_omega", LKJCholesky(
#             m_sample_categories, np.ones(1)))
#         c = jax.numpy.matmul(jax.numpy.diag(theta.sqrt()), L_omega)
#         print("L_omega", L_omega.shape)
#         # TODO: understand this note from Pyro code
#         # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)

#         # Vector of expectations
#         probs = MultivariateNormal(pd_hat, scale_tril=L_omega)

#     with pyro.plate("data", data.shape[0]):
#         z = pyro.sample("z", Categorical(mix_weights(beta)))
#         return pyro.sample("obs", Multinomial(probs=probs[z]), obs=data)


def infer(model_to_run, data, n_cases: np.array, n_ctrls: int, max_K: int, max_tree_depth: int, jit_model_args: bool,
          num_warmup: int, num_samples: int, num_chains: int, chain_method: str, hmcecs_blocks: Optional[int] = 0) -> MCMC:
    kernel = NUTS(model_to_run)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, jit_model_args=jit_model_args, num_chains=num_chains, chain_method=chain_method)
    mcmc.run(data, n_cases, n_ctrls, max_K)
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


def run(run_params, folder_prefix: str = "") -> Tuple[MCMC, Tuple]:
    # mcmc = infer(**run_params)
    optim = Adam({"lr": 0.05})
    guide = AutoNormal(model)

    svi = SVI(model, guide, optim, loss=TraceGraph_ELBO())
    losses = []
    def train(num_iterations):
        pyro.clear_param_store()
        for j in tqdm(range(num_iterations)):
            loss = svi.step(run_params['data'], run_params['n_cases'], run_params['n_ctrls'], run_params['max_K'])
            losses.append(loss)

    def truncate(alpha, weights):
        threshold = alpha**-1 / 100.
        # true_centers = centers[weights > threshold]
        true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
        return true_weights

    # alpha = 0.1
    train(1000)

    # We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
    # Bayes_Centers_01, Bayes_Weights_01 = truncate(alpha, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

    # inferred_params = get_inferred_params(mcmc)
    # suffix = uuid.uuid4()
    # folder = datetime.datetime.now().strftime('%h-%d-%y-%H-%M-%S') + f"_{suffix}"
    # if folder_prefix:
    #     folder = f"{folder_prefix}_{folder}"

    # os.mkdir(folder)

    # with open(f"{folder}/inferred_params.pickle", "wb") as f:
    #     cloudpickle.dump(inferred_params, f)

    # with open(f"{folder}/mcmc.pickle", "wb") as f:
    #     mcmc_to_save = copy.deepcopy(mcmc)
    #     mcmc_to_save.sampler._sample_fn = None  # pylint: disable=protected-access
    #     mcmc_to_save.sampler._init_fn = None  # pylint: disable=protected-access
    #     mcmc_to_save.sampler._constrain_fn = None  # pylint: disable=protected-access
    #     mcmc_to_save._cache = {}  # pylint: disable=protected-access
    #     cloudpickle.dump(mcmc_to_save, f)

    # with open(f"{folder}/sim_data.pickle", "wb") as f:
    #     cloudpickle.dump(sim_data, f)

    # with open(f"{folder}/run_params.pickle", "wb") as f:
    #     cloudpickle.dump(run_params, f)

    # return mcmc, inferred_params

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
        run_params = cloudpickle.load(file)

    sim_data = None
    with open(os.path.join(folder, "sim_data.pickle"), 'rb') as file:
        sim_data = cloudpickle.load(file)
        
    return run_params, sim_data

