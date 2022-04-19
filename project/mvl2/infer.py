from typing import Any, Tuple, Optional, Literal, Iterable
from collections.abc import Iterable as IterableCollection
import datetime
import os
import copy
import multiprocessing
import uuid
from typing import List

import dill

import jax
from jax import random
from jax.nn import softmax
import jax.numpy as jnp

import numpy as np
import numpyro
from numpyro.distributions import *
from numpyro.infer import MCMC, NUTS
from numpyro.contrib.funsor.infer_util import config_enumerate

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
    if len(beta.shape) == 3:
        return jnp.stack(list(mix_weights_one_chain(beta[i]) for i in range(beta.shape[0])))
    else:
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
def model_with_gamma_prior_alpha(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(pd_hat, 1).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)


def model_with_dirichlet_prior_alpha2(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("concentrations_plate", 1):
        concentrations = numpyro.sample("dirichlet_concentration", Dirichlet(get_pdhat(n_cases, n_ctrls)))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

# This model appears to work best
def model_with_gamma_prior_alpha3(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))
    
    with numpyro.plate("concentrations_plate", 1):
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(get_pdhat(n_cases, n_ctrls), 1/k_hypotheses))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

# This is the preferred model
def model_with_gamma_prior_alpha4(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("concentrations_plate", 1):
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(get_pdhat(n_cases, n_ctrls), 1))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model_with_gamma_prior_alpha5(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("concentrations_plate", 1):
        prevalence_in_sample_estimate = get_pdhat(n_cases, n_ctrls)
        prevalence_in_sample_estimate_std = prevalence_in_sample_estimate.std(0)
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(prevalence_in_sample_estimate, prevalence_in_sample_estimate_std))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model_with_gamma_prior_alpha6(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("concentrations_plate", 1):
        marginal_probs = data / data.sum(1)[:,np.newaxis]
        empirical_prevalence_estimate = marginal_probs.mean(0)
        std = marginal_probs.std(0)
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(empirical_prevalence_estimate, std))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)


def model_with_gamma_prior_alpha7(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    with numpyro.plate("concentrations_plate", 1):
        marginal_probs = data / data.sum(1)[:,np.newaxis]
        empirical_prevalence_estimate = marginal_probs.mean(0)
        std = marginal_probs.std(0)

        moment_methods_shape = empirical_prevalence_estimate**2 / std**2
        moment_methods_rate = empirical_prevalence_estimate / std
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(moment_methods_shape, moment_methods_rate))

    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)


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

def model_with_gamma_prior_alpha2(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        concentrations = numpyro.sample("dirichlet_concentration", Gamma(pd_hat, 1/k_hypotheses).to_event(1))
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

@config_enumerate
def model_enumerate(data, n_cases: np.array, n_ctrls: int, k_hypotheses: int, alpha: float = .05):
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha / k_hypotheses))

    pd_hat = get_pdhat(n_cases, n_ctrls)
    with numpyro.plate("prob_plate", k_hypotheses):
        probs = numpyro.sample("probs", Dirichlet(pd_hat))

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)).mask(False))
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
        c = jnp.matmul(jnp.diag(theta.sqrt()), L_omega)
        print("L_omega", L_omega.shape)
        # TODO: understand this note from Pyro code
        # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)

        # Vector of expectations
        probs = MultivariateNormal(pd_hat, scale_tril=L_omega)

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)


def infer(random_key: random.PRNGKey, model_to_run, data, n_cases: np.array, n_ctrls: int, max_K: int, max_tree_depth: int, jit_model_args: bool,
          num_warmup: int, num_samples: int, num_chains: int, chain_method: str, target_accept_prob: float = 0.8, hmcecs_blocks: Optional[int] = 0, alpha=.05, extra_fields = (),
          thinning: int = 1) -> MCMC:
    kernel = NUTS(model_to_run, target_accept_prob=target_accept_prob)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, jit_model_args=jit_model_args, num_chains=num_chains, chain_method=chain_method, thinning=thinning)
    mcmc.run(random_key, data, n_cases, n_ctrls, max_K, alpha, extra_fields=extra_fields)
    mcmc.print_summary()
    return mcmc

def get_inferred_params(mcmc: MCMC) -> Tuple[Any, Any]:
    posterior_samples = mcmc.get_samples()
    print(posterior_samples)
    beta = posterior_samples['beta']

    weights = mix_weights(beta)

    print("inferred stick-breaking weights mean: ", weights.mean(0))
    print("inferred stick-breaking weights stdd: ", weights.std(0))

    return posterior_samples, weights


def run(random_key, run_params, pickle_results: bool = True, folder_prefix: str = "") -> Tuple[MCMC, Tuple]:
    mcmc = infer(random_key, **run_params)
    inferred_params = get_inferred_params(mcmc)

    if not pickle_results:
        return mcmc, inferred_params

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

    with open(f"{folder}/run_params.pickle", "wb") as f:
        dill.dump(run_params, f)

    return mcmc, inferred_params

# TODO: implement pmap version that uses a chunk size that is the jax.device_count()
# rkeys= random.split(random_key, 4)
# kernel = NUTS(model=model, target_accept_prob=.8)
# mcmc = MCMC(kernel, num_warmup=2000, num_samples=4000, jit_model_args=False, chain_method='sequential', progress_bar=False)

# res = vmap(lambda rkey: mcmc.run(random_key, stat_data_dave, n_cases, n_ctrls, 4, .05))(rkeys)
def run_until_enough(random_key, run_params, target_number_of_chains=4, acceptance_threshold=.7, max_attempts=10):
    """
        May return more than target_number_of_chains when in parallel mode
    """
    accepted = []
    n_attempts = 0
    rkeys = random.split(random_key, max_attempts)

    while len(accepted) < target_number_of_chains and n_attempts < max_attempts:
        r_mcmc, _ = run(rkeys[n_attempts], run_params)
        accept_prob = r_mcmc.get_extra_fields()['accept_prob'].mean(0)

        if accept_prob >= acceptance_threshold:
            accepted.append(r_mcmc)
        
        n_attempts += 1

    return accepted

# TODO: make generic in sample site names
@jax.jit
def ordered_statistics(runs_mcmc, order: Iterable[int] = None): 
    # Simple ordering procedure
    # We'll modify this to not argsort, but instead permute and maximize likelihood
    all_weights_ordered = []
    all_beta_ordered = []
    all_probs_ordered = []
    dirichlet_concentrations = []

    make_order = False
    if order is None:
        make_order = True

    for mr in runs_mcmc:
        posterior_probs = mr.get_samples()
        probs = posterior_probs['probs']
        betas = posterior_probs['beta']
        weights = np.array(mix_weights(posterior_probs['beta']))

        probs_ordered = []
        weights_ordered = []
        betas_ordered = []

        if make_order:
            order = np.argsort(weights.mean(0))[::-1]

        print("order", order)

        for prob_chain in probs:
            probs_ordered.append(np.array(prob_chain[order]))

        for weight_chain in weights:
            weights_ordered.append(np.array(weight_chain[order]))

        for beta_chain in betas:
            betas_ordered.append(np.array(beta_chain[order]))

        weights_ordered = np.array(weights_ordered)
        probs_ordered = np.array(probs_ordered)

        all_weights_ordered.append(weights_ordered)
        all_probs_ordered.append(probs_ordered)
        all_beta_ordered.append(betas_ordered)

        if 'dirichlet_concentration' in posterior_probs:
            concentration_ordered = []
            for conc_chain in posterior_probs['dirichlet_concentration']:
                concentration_ordered.append(np.array(conc_chain[order]))
            dirichlet_concentrations.append(np.array(concentration_ordered))

    all_weights_ordered = np.stack(all_weights_ordered)
    all_probs_ordered = np.stack(all_probs_ordered)
    all_beta_ordered = np.stack(all_beta_ordered)

    if not dirichlet_concentrations:
        dirichlet_concentrations = None
    else:
        dirichlet_concentrations = np.stack(dirichlet_concentrations)

    return all_weights_ordered, all_probs_ordered, all_beta_ordered, dirichlet_concentrations

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

