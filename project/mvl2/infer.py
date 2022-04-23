from typing import Any, Tuple, Optional, Literal, Iterable
from collections.abc import Iterable as IterableCollection
import datetime
import os
import copy
import multiprocessing
import uuid
from typing import List

import dill

from jax import random
from jax.nn import softmax
import jax.numpy as jnp

import numpy as np
import numpyro
from numpyro.distributions import *
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.funsor.infer_util import config_enumerate

import pandas as pd

import sigfig

numpyro.set_host_device_count(multiprocessing.cpu_count())

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

def mix_weights_one_chain(beta: jnp.array):
    beta_cumprod = (1 - beta).cumprod(-1)
    return jnp.pad(beta, (0, 1), constant_values=1) * jnp.pad(beta_cumprod, (1, 0), constant_values=1)

def mix_weights(beta: jnp.array):
    # multiple chains
    if len(beta.shape) > 1:
        return jnp.stack(list(mix_weights_one_chain(beta[i]) for i in range(beta.shape[0])))
    
    return mix_weights_one_chain(beta)
    

# Covariates needed
# Sex of the individual
# parent of origin would be important
def model_with_halfnormal_alpha(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        concentrations = numpyro.sample("dirichlet_concentration", HalfNormal(pd_hat).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model_with_uniform_alpha(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        concentrations = numpyro.sample("dirichlet_concentration", Uniform(pd_hat).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model_with_dirichlet_prior_alpha(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        concentrations = numpyro.sample("dirichlet_concentration", Dirichlet(pd_hat))
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

########################## Gamma ppooled 
# https://arxiv.org/pdf/1708.08177.pdf
def model_with_gamma_prior_alpha8(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("concentrations_plate", 1):
        empirical_prevalence_estimate = data.mean(0)
        std = data.std(0)

        moment_methods_shape = empirical_prevalence_estimate**2 / std**2
        moment_methods_rate = empirical_prevalence_estimate / std
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(moment_methods_shape, moment_methods_rate))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(pd_hat))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def infer(random_key: random.PRNGKey, model_to_run, data, n_cases: np.array, n_ctrls: int, max_K: int, max_tree_depth: int, jit_model_args: bool,
          num_warmup: int, num_samples: int, num_chains: int, chain_method: str, target_accept_prob: float = 0.8, alpha=.05, extra_fields = (),
          thinning: int = 1) -> MCMC:
    kernel = NUTS(model_to_run, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, jit_model_args=jit_model_args, num_chains=num_chains, chain_method=chain_method, thinning=thinning)
    mcmc.run(random_key, data, n_cases, n_ctrls, max_K, alpha, extra_fields=extra_fields)
    mcmc.print_summary()
    return mcmc

@np.vectorize
def round_it(x, sig_figs:int = 2):
    return sigfig.round(x, sig_figs)

def run(random_key, run_params, pickle_results: bool = True, folder_prefix: str = "") -> Tuple[MCMC, Tuple]:
    mcmc = infer(random_key, **run_params)

    if not pickle_results:
        return mcmc

    suffix = uuid.uuid4()
    folder = datetime.datetime.now().strftime('%h-%d-%y-%H-%M-%S') + f"_{suffix}"

    if folder_prefix:
        folder = f"simulations/{folder_prefix}_{folder}"

    os.mkdir(folder)

    with open(f"{folder}/samples.pickle", "wb") as f:
        dill.dump(mcmc.get_samples(), f)

    with open(f"{folder}/mcmc.pickle", "wb") as f:
        mcmc_to_save = copy.deepcopy(mcmc)
        mcmc_to_save.sampler._sample_fn = None  # pylint: disable=protected-access
        mcmc_to_save.sampler._init_fn = None  # pylint: disable=protected-access
        mcmc_to_save.sampler._constrain_fn = None  # pylint: disable=protected-access
        mcmc_to_save._cache = {}  # pylint: disable=protected-access
        dill.dump(mcmc_to_save, f)

    with open(f"{folder}/run_params.pickle", "wb") as f:
        dill.dump(run_params, f)

    return mcmc

# TODO: should this be using 'mean_accept_prob' to test against acceptance_threshold?
# accept_prob gives higher variances, means appear very similar, but are there pathological cases?
def run_until_enough(random_key, run_params, target_number_of_chains=4, acceptance_threshold=.7, max_attempts=10, pickle_results:bool = False):
    """
        May return more than target_number_of_chains when in parallel mode
    """
    accepted = []
    n_attempts = 0
    rkeys = random.split(random_key, max_attempts)

    while len(accepted) < target_number_of_chains and n_attempts < max_attempts:
        r_mcmc = run(rkeys[n_attempts], run_params, pickle_results=pickle_results)
        accept_prob = r_mcmc.get_extra_fields()['accept_prob'].mean(0)

        if accept_prob >= acceptance_threshold:
            accepted.append(r_mcmc)
        
        n_attempts += 1

    return accepted

# TODO: improve selection criteria by adding n_eff mean/std, r_hat mean/std, and improve potential_energy use
def select_best_chain(runs_mcmc):
    """
    critera: List[Tuple[str, bool]]
      The key, value pair. Values are boolean; True indicates prefer maximum (descending sort)
    """
    
    res = []
    for i, run in enumerate(runs_mcmc):
        fields = run.get_extra_fields()
        diverging = np.count_nonzero(fields['diverging'])
        accept_prob = fields['accept_prob']
        accept_prob_mean = accept_prob.mean()
        accept_prob_std = accept_prob.std()
        potential_energy = fields['potential_energy'][-1]

        res.append(np.array([diverging, accept_prob_mean, accept_prob_std, potential_energy, i]))

    res = pd.DataFrame(res, columns=['diverging', 'accept_prob_mean', 'accept_prob_std', 'potential_energy', 'mcmc_index'])
    res = res.sort_values(by=['diverging', 'accept_prob_mean', 'accept_prob_std', 'potential_energy'], ascending=[True, False, True, True])

    best_index = res.loc[0,'mcmc_index']

    return runs_mcmc[int(best_index)], res

def get_parameters(mcmc_run: MCMC):
    posterior_probs = mcmc_run.get_samples()
    weights = np.array(mix_weights(posterior_probs['beta']))

    return weights, posterior_probs['probs'], posterior_probs['beta'], posterior_probs.get('dirichlet_concentration')

# TODO: Is it safe to assume hypotheses correspond to maximizing penetrance?
def get_assumed_order_for_2(probs):
    """
    Infer the order of hypotheses for 2 conditions and 4 channels: ctrls, cases1, cases2, cases_both
    """
    hypotheses = {}

    probs_mean_rounded = round_it(probs.mean(0))
    probs_mean_rounded_df = pd.DataFrame(probs_mean_rounded, columns=['P(~D|V,H)', 'P(D1|V,H)', 'P(D2|V,H)', 'P(D12|V,H)'])

    h0 = probs_mean_rounded_df['P(~D|V,H)'].idxmax()
    hypotheses[h0] = 'H0'
    h1 = (probs_mean_rounded_df['P(D1|V,H)'] - probs_mean_rounded_df['P(D2|V,H)']).idxmax() #(case1 > case2)
    hypotheses[h1] = 'H1'
    h2 = (probs_mean_rounded_df['P(D2|V,H)'] - probs_mean_rounded_df['P(D1|V,H)']).idxmax() #(case2 > case1)
    hypotheses[h2] = 'H2'
    h12 = (probs_mean_rounded_df['P(D12|V,H)']).idxmax()
    hypotheses[h12] = 'H12'

    probs_mean_rounded_df.index = [hypotheses[k] for k in sorted(hypotheses.keys())]
    
    return np.array([h0, h1, h2, h12]), probs_mean_rounded_df

# this will only work for well-separated values
# instead, we should be ordering by both probs and weight, maybe likelihood?
def ordered_statistics(runs_mcmc: Iterable[MCMC], order: Iterable[int] = None): 
    # Simple ordering procedure
    # We'll modify this to not argsort, but instead permute and maximize likelihood
    all_weights_ordered = []
    all_probs_ordered = []
    all_dirichlet_concentrations = []
    all_betas = []

    make_order = False
    if order is None:
        make_order = True

    for mr in runs_mcmc:
        weights, probs, beta, dirichlet_concentrations = get_parameters(mr)
        
        # I am yet sure how to order these
        all_betas.append(beta)

        if make_order:
            order = np.argsort(weights.mean(0))[::-1]

        weights_ordered = np.take_along_axis(weights, np.expand_dims(order, axis=(0)), axis=1)
        probs_ordered = np.take_along_axis(probs, np.swapaxes(np.expand_dims(order, axis=(0, 1)), 2, 1), axis=1)

        all_weights_ordered.append(weights_ordered)
        all_probs_ordered.append(probs_ordered)

        if isinstance(dirichlet_concentrations, np.ndarray) and dirichlet_concentrations.size > 0:
            conc_ordered = np.take_along_axis(dirichlet_concentrations, np.expand_dims(order, axis=(0)), axis=1)
            all_dirichlet_concentrations.append(np.array(conc_ordered))

    all_weights_ordered = np.stack(all_weights_ordered)
    all_probs_ordered = np.stack(all_probs_ordered)

    if not all_dirichlet_concentrations:
        all_dirichlet_concentrations = None
    else:
        all_dirichlet_concentrations = np.stack(all_dirichlet_concentrations)

    all_betas = np.stack(all_betas)

    if len(runs_mcmc) == 1:
        return all_weights_ordered[0], all_probs_ordered[0], all_betas[0], (all_dirichlet_concentrations[0] if all_dirichlet_concentrations else None)

    return all_weights_ordered, all_probs_ordered, all_betas, all_dirichlet_concentrations



def get_statistics_permutations(runs_mcmc, K=4):
    # K is the number of components
    # we will have 1 weight per component
    # and each component will have a likelihood array, so probability array will be K x n_sample_categories
    for order in permutations(K):
        yield ordered_statistics(runs_mcmc, order)

# def order_by_maximum_likelihood_ratio(runs_mcmc):
#     for mr in runs_mcmc:
#         posterior_probs = mr.get_samples()
#         probs = posterior_probs['probs']
#         # betas = posterior_probs['beta']
#         weights = np.array(mix_weights(posterior_probs['beta']))

#         probs_ordered = []
#         weights_ordered = []
#         for prob_mcmc_sample in probs:
#             probs_ordered.append(prob_mcmc_sample[order])

#         for weight in weights:
#             weights_ordered.append(weight[order])

def permutations(n):
    # https://stackoverflow.com/questions/64291076/generating-all-permutations-efficiently
    a = np.zeros((np.math.factorial(n), n), np.uint8)
    f = 1
    for m in range(2, n+1):
        b = a[:f, n-m+1:]      # the block of permutations of range(m-1)
        for i in range(1, m):
            a[i*f:(i+1)*f, n-m] = i
            a[i*f:(i+1)*f, n-m+1:] = b + (b >= i)
        b += 1
        f *= m
    return a


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
        
    return run_params

