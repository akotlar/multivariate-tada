from typing import Any, Callable, Tuple, Optional, Literal, Iterable, Union
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
import numpy.typing as npt
import numpyro
from numpyro.distributions import *
from numpyro.infer import MCMC, NUTS
from numpyro.distributions.conjugate import _log_beta_1

from scipy.stats import norm, binom, poisson

Inv_Cumulative_Normal = norm.ppf

import pandas as pd

import sigfig
import jax

class TruncatedMultinomialProbs(MultinomialProbs):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, *args, k: float = -500., **kwargs):
        super().__init__(*args, **kwargs)
        self.k = float(k)

    def log_prob(self, value):
        ll = super().log_prob(value)
        return jnp.where(jnp.isnan(ll) | (ll < self.k), self.k, ll)

class ZeroInflatedTruncatedMultinomial(MultinomialProbs):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, *args, gate_n: int, gate_pop_af: float, k: float = -500, **kwargs):
        super().__init__(*args, **kwargs)
        prob_zero = jax.scipy.stats.poisson.pmf(0, gate_n * gate_pop_af)
        
        self.gate = jnp.expand_dims(prob_zero, 1)

        self.k = k

    def log_prob(self, value):
        ll = super().log_prob(value)
        ll = jnp.where(value.T == 0, jnp.log(self.gate + jnp.exp(ll)), jnp.log1p(-self.gate) + ll)

        ll = jnp.where(jnp.isnan(ll) | (ll < self.k), self.k, ll)

        return ll

    # @constraints.dependent_property(is_discrete=True, event_dim=0)
    # def support(self):
    #     return self.base_dist.support

    @numpyro.distributions.util.lazy_property
    def mean(self):
        raise NotImplementedError
        # return (1 - self.gate) * self.base_dist.mean

    @numpyro.distributions.util.lazy_property
    def variance(self):
        raise NotImplementedError
        # return (1 - self.gate) * (
        #     self.base_dist.mean**2 + self.base_dist.variance
        # ) - self.mean**2

class TruncatedDirichlet(Dirichlet):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, *args, k: float = -500, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = jnp.array([k]*self.concentration.shape[0])

    def log_prob(self, value):
        ll = super().log_prob(value)

        ll = jnp.where(jnp.isnan(ll) | (ll < self.k), self.k, ll)

        return ll

class ZeroInflatedTruncatedDirichlet(Dirichlet):
    arg_constraints = {
        "probs": constraints.simplex,
        "total_count": constraints.nonnegative_integer,
    }

    def __init__(self, *args, gate_n: float, gate_pop_af: float, data: jnp.array, k: float = -500., **kwargs):
        super().__init__(*args, **kwargs)
        prob_zero = jax.scipy.stats.poisson.pmf(0, gate_n * gate_pop_af)
        
        self.gate = jnp.expand_dims(prob_zero, 1)
        self.k = float(k)
        self.data = data

    def log_prob(self, value):
        ll = super().log_prob(value)
        ll = jnp.where(self.data == 0, jnp.log(self.gate + jnp.exp(ll)), jnp.log1p(-self.gate) + ll)

        ll = jnp.where(jnp.isnan(ll) | (ll < self.k), self.k, ll)

        return ll

numpyro.set_host_device_count(multiprocessing.cpu_count())
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

########################## Gamma prior on dirichlet concentrations 
# https://arxiv.org/pdf/1708.08177.pdf
def method_moments_estimator_gamma_shape_rate(empirical_prevalence_estimate: ArrayLike = None, data: ArrayLike = None):
    assert data is not None or empirical_prevalence_estimate is not None

    if empirical_prevalence_estimate is None:
        empirical_prevalence_estimate = data.mean(0)

    sd = data.std(0)

    moment_methods_shape = empirical_prevalence_estimate**2 / sd**2
    moment_methods_rate = empirical_prevalence_estimate / sd

    return moment_methods_shape, moment_methods_rate

def modelLKJ(data: ArrayLike = None, k_hypotheses: int = 4, alpha: float = .05,
                                  shared_dirichlet_prior: bool = False,
                                  gamma_shape: Union[float, ArrayLike] = None,
                                  gamma_rate: Union[float, ArrayLike] = None):
    d = data.shape[1]
    # # options = dict(dtype=data.dtype, device=data.device)
    # # Vector of variances for each of the d variables
    # theta = numpyro.sample("theta", HalfCauchy(jnp.ones(d)))
    # # Lower cholesky factor of a correlation matrix
    # concentration = jnp.ones(())  # Implies a uniform distribution over correlation matrices
    # print('concentration', concentration)
    # L_omega = numpyro.sample("L_omega", LKJCholesky(d, concentration))
    # print('theta', theta)
    # print('L_omega', L_omega)
    # # Lower cholesky factor of the covariance matrix
    # L_Omega = jnp.matmul(jnp.diag(jnp.sqrt(theta)), L_omega)
    # # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)
    # print('L_Omega', L_Omega)


    # # Vector of expectations
    # mu = np.zeros(d)

    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("prob_plate", k_hypotheses):
        # options = dict(dtype=data.dtype, device=data.device)
        # Vector of variances for each of the d variables
        # theta = numpyro.sample("theta", HalfCauchy(jnp.ones(d)))
        # # Lower cholesky factor of a correlation matrix
        # L_omega = numpyro.sample("L_omega", LKJCholesky(d, .5))
        # # print('theta', theta)
        # # print('L_omega', L_omega)
        # # Lower cholesky factor of the covariance matrix
        # L_Omega = jnp.matmul(jnp.diag(jnp.sqrt(theta)), L_omega)
        # # For inference with SVI, one might prefer to use torch.bmm(theta.sqrt().diag_embed(), L_omega)
        # # print('L_Omega', L_Omega)
        # # print('cov_est', cov_est)
        # mu = np.zeros(d)if gamma_shape is None:
        gamma_shape, gamma_rate = method_moments_estimator_gamma_shape_rate(data=data)

        scale = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate))

        # d = Gamma(1, 1).expand([data.shape[1]]).to_event(1)
        # scale = numpyro.sample('scale', d)

        # Sample the correlation matrix.
        d = LKJCholesky(data.shape[1], .25)
        cholesky_corr = numpyro.sample('cholesky_corr', d)

        # Evaluate the Cholesky decomposition of the covariance matrix.
        cholesky_cov = cholesky_corr * jnp.sqrt(scale[:, None])
        # print('cholesky_cov', cholesky_cov)
        logtheta = numpyro.sample("logtheta", MultivariateNormal(0, scale_tril=cholesky_cov))
        probs = jax.nn.softmax(logtheta, -1)
        # theta = jnp.exp(logtheta)
        # print('probs', probs)

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def model(data: ArrayLike, num_data: int, N: ArrayLike, prior_pop_af: float,
                                  dirichlet_min_ll: float = -20,
                                  multinomial_min_ll: float = -200,
                                  k_hypotheses: int = 4, alpha: float = .05,
                                  shared_dirichlet_prior: bool = False,
                                  gamma_shape: Union[float, ArrayLike] = None,
                                  gamma_rate: Union[float, ArrayLike] = None):
    """
        Parameters
        ----------
        shared_dirichlet_prior: bool
            Whether each component should share the same Gamma prior.
            This results in fewer parameters to estimate, and may perform better when the genetic architecture allows for it.
    """
    with numpyro.plate("beta_plate", k_hypotheses-1):
        alpha = numpyro.deterministic('alpha', get_alpha(k_hypotheses, alpha))
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("prob_plate", k_hypotheses):
        if gamma_shape is None:
            assert gamma_rate is None and data is not None
            gamma_shape, gamma_rate = method_moments_estimator_gamma_shape_rate(data=data)
        assert gamma_shape is not None and gamma_rate is not None
        gamma_rate = numpyro.deterministic('gamma_rate', gamma_rate)
        gamma_shape = numpyro.deterministic('gamma_shape', gamma_shape)

        if shared_dirichlet_prior:
            concentrations = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate))
        else:
            concentrations = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate).to_event(1))

        probs = numpyro.sample('probs', TruncatedDirichlet(concentrations, k=dirichlet_min_ll))

    with numpyro.plate("data", num_data):
        pz = numpyro.deterministic("pz", mix_weights(beta))
        z = numpyro.sample("z", Categorical(pz), infer={"enumerate": "parallel"})

        return numpyro.sample("obs", ZeroInflatedTruncatedMultinomial(probs=probs[z], gate_n=N, gate_pop_af=prior_pop_af, k=multinomial_min_ll), obs=data)

def modelPoisson(data: ArrayLike = None, k_hypotheses: int = 4, alpha: float = .05,
                                  shared_dirichlet_prior: bool = False,
                                  gamma_shape: Union[float, ArrayLike] = None,
                                  gamma_rate: Union[float, ArrayLike] = None):
    """
        Parameters
        ----------
        shared_dirichlet_prior: bool
            Whether each component should share the same Gamma prior.
            This results in fewer parameters to estimate, and may perform better when the genetic architecture allows for it.
    """
    # x = numpyro.sample('x', ImproperUniform(constraints.ordered_vector, (), event_shape=(4,)))
    # print("x", x)
    # alpha = numpyro.param("alpha", alpha)
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("prob_plate", k_hypotheses):
        if gamma_shape is None:
            assert gamma_rate is None and data is not None
            gamma_shape, gamma_rate = method_moments_estimator_gamma_shape_rate(data)
        assert gamma_shape is not None and gamma_rate is not None

        if shared_dirichlet_prior:
            concentrations = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate))
        else:
            concentrations = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate).to_event(1))
        # probs = numpyro.sample("probs", Dirichlet(concentrations))
        rate = numpyro.sample('rate', Gamma(gamma_shape, gamma_rate).to_event(1))
        concentration = numpyro.sample('concentration', Gamma(gamma_shape, gamma_rate).to_event(1))
    lmbda = numpyro.sample("lambda", Gamma([[gamma_shape,gamma_shape,gamma_shape,gamma_shape]], [[gamma_rate,gamma_rate,gamma_rate,gamma_rate]]))
    # t =
    # t = jnp.array([[0,1,2,3],[0,1,2,3],[0,1,2,3],[0,1,2,3]])

    with numpyro.plate("data", data.shape[0]):
        pz = numpyro.deterministic("pz", mix_weights(beta))
        z = numpyro.sample("z", Categorical(pz), infer={"enumerate": "parallel"})
        print('lmbda',lmbda.shape)
        print('data', data.shape)
        print('data', data[0])
        print('lmbda[z]', lmbda[z].shape)
        # print('probs', probs)
        return numpyro.sample("obs", Poisson(lmbda[z]), obs=data)

def modelFinite(data: ArrayLike = None, k_hypotheses: int = 4, alpha: float = .05,
                                  shared_dirichlet_prior: bool = False,
                                  gamma_shape: Union[float, ArrayLike] = None,
                                  gamma_rate: Union[float, ArrayLike] = None):
    """
        Parameters
        ----------
        shared_dirichlet_prior: bool
            Whether each component should share the same Gamma prior.
            This results in fewer parameters to estimate, and may perform better when the genetic architecture allows for it.
    """
    # x = numpyro.sample('x', ImproperUniform(constraints.ordered_vector, (), event_shape=(4,)))
    # print("x", x)
    # alpha = numpyro.param("alpha", alpha)
    with numpyro.plate("beta_plate", k_hypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("prob_plate", k_hypotheses):
        if gamma_shape is None:
            assert gamma_rate is None and data is not None
            gamma_shape, gamma_rate = method_moments_estimator_gamma_shape_rate(data)
        assert gamma_shape is not None and gamma_rate is not None

        if shared_dirichlet_prior:
            concentrations = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate))
        else:
            concentrations = numpyro.sample("dirichlet_concentration", Gamma(gamma_shape, gamma_rate).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(concentrations))

    with numpyro.plate("data", data.shape[0]):
        pz = numpyro.deterministic("pz", mix_weights(beta))
        # z = numpyro.sample("z", Categorical(pz), infer={"enumerate": "parallel"})
        return numpyro.sample("obs", MixtureSameFamily(Categorical(pz), Multinomial(probs=probs)), obs=data)

def get_alpha(max_K: int, base_alpha: float = .05):
    return base_alpha/max_K

def infer(random_key: random.PRNGKey, data: ArrayLike,
          model: Callable = model,
          jit_model_args: bool = False, num_warmup: int = 2000, num_samples: int = 4000, num_chains: int = 1, chain_method: str = 'parallel', 
          target_accept_prob: float = 0.8, max_tree_depth: int = 10,
          thinning: int = 1, print_diagnostics: bool = True, extra_fields: Tuple['str'] = ("potential_energy", "energy", "accept_prob", "mean_accept_prob"), **kwargs) -> MCMC:
    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    """
    "max_tree_depth": values less than 10 give very bad results
    """
    print("model chosen:", model)
    assert max_tree_depth >= 10

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, jit_model_args=jit_model_args, num_chains=num_chains, chain_method=chain_method, thinning=thinning)
    mcmc.run(random_key, data, extra_fields=extra_fields, **kwargs)

    if print_diagnostics:
        mcmc.print_summary()
        weights, probs, _, _ = get_parameters(mcmc)
        print("weights.mean(0)", weights.mean(0))
        print("probs.mean(0)", probs.mean(0))
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

# TODO: get this from last state, or predictive?
def get_parameters(mcmc_run: MCMC):
    posterior_probs = mcmc_run.get_samples()
    weights = np.array(mix_weights(posterior_probs['beta']))

    return weights, posterior_probs.get('probs'), posterior_probs['beta'], posterior_probs.get('dirichlet_concentration')

# TODO: Is it safe to assume hypotheses correspond to maximizing penetrance?
# def get_assumed_order_for_2(probs, data_columns=['unaffected', 'affected1', 'affected2', 'affected12']):
#     """
#     Infer the order of hypotheses for 2 conditions and 4 channels: ctrls, cases1, cases2, cases_both
#     """
#     hypotheses = {}

#     mapping= {
#         'unaffected': 'P(~D|V,H)',
#         'affected1': 'P(D1|V,H)',
#         'affected2': 'P(D2|V,H)',
#         'affected12': 'P(D12|V,H)'
#     }
#     print("[mapping[x] for x in data_columns]", [mapping[x] for x in data_columns])
#     probs_mean_rounded = round_it(probs.mean(0), 3)
#     display("probs_mean_rounded", probs_mean_rounded)
#     probs_mean_rounded_df = pd.DataFrame(probs_mean_rounded, columns=[mapping[x] for x in data_columns])
#     print("probs_mean_rounded_df", probs_mean_rounded_df)
#     print(probs_mean_rounded_df['P(~D|V,H)'])
#     h0 = probs_mean_rounded_df['P(~D|V,H)'].idxmax()
#     hypotheses[h0] = 'H0'
#     h1 = (probs_mean_rounded_df['P(D1|V,H)'] - probs_mean_rounded_df['P(D2|V,H)']).idxmax() #(case1 > case2)
#     print("h1", h1)
#     hypotheses[h1] = 'H1'
#     h2 = (probs_mean_rounded_df['P(D2|V,H)'] - probs_mean_rounded_df['P(D1|V,H)']).idxmax() #(case2 > case1)
#     print("h2", h2)
#     hypotheses[h2] = 'H2'
#     h12 = (probs_mean_rounded_df['P(D12|V,H)']).idxmax()
#     print('h12', h12)
#     hypotheses[h12] = 'H12'

#     print('hypotheses', hypotheses)

#     probs_mean_rounded_df.index = [hypotheses[k] for k in probs_mean_rounded_df.index]
    
#     return np.array([h0, h1, h2, h12]), probs_mean_rounded_df

# TODO: use prevalences to infer order
def get_assumed_order_for_2(probs: np.ndarray, data_columns: List[str] = ['unaffected', 'affected1', 'affected2', 'affected12'], prevalences: np.ndarray = None, data: np.ndarray = None, h2_first = True):
    """
    Infer the order of hypotheses for 2 conditions and 4 channels: ctrls, cases1, cases2, cases_both
    prevalences: Iterable[Union[int, float]]
        The list of prevalences for each of the condition columns (ex: ctrls, cases1, cases2, cases for both). Should be in the same order as the data columns
        The last element of the prevalences array should be comorbidity, if known
    h2_first: Bool
        If H2 (Unaffected_Affected) is in the column indexed 1 (2nd column, first being Unaffected_Unaffected)
    """
    hypotheses = {}

    mapping= {
        'unaffected': 'P(~D|V,H)',
        'affected1': 'P(D1|V,H)',
        'affected2': 'P(D2|V,H)',
        'affected12': 'P(D12|V,H)',
        'outlier': 'P(outlier)'
    }

    orig_mapping = {}

    probs_mean = probs.mean(0)#round_it(probs.mean(0), 4)
    probs_mean_df = pd.DataFrame(probs_mean, dtype='float32', columns=[mapping[x] for x in data_columns])

    h0 = probs_mean_df['P(~D|V,H)'].idxmax()
    orig_mapping[h0] = 0
    hypotheses[h0] = 'H0'
    h1 = (probs_mean_df['P(D1|V,H)'] - probs_mean_df['P(D2|V,H)']).idxmax() #(case1 > case2)
    hypotheses[h1] = 'H1'

    if h2_first:
        orig_mapping[h1] = 2
    else:
        orig_mapping[h1] = 1

    h2 = (probs_mean_df['P(D2|V,H)'] - probs_mean_df['P(D1|V,H)']).idxmax() #(case2 > case1)
    hypotheses[h2] = 'H2'

    if h2_first:
        orig_mapping[h2] = 1
    else:
        orig_mapping[h2] = 2

    h12 = (probs_mean_df['P(D12|V,H)']).idxmax()
    hypotheses[h12] = 'H12'
    orig_mapping[h12] = 3

    probs_mean_df.index = [hypotheses[k] for k in probs_mean_df.index]
    
    return np.array([h0, h1, h2, h12]), probs_mean_df, orig_mapping

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

def get_assumed_order_bivariate(probs: np.ndarray, data: np.ndarray = None, data_columns: List[str] = ['unaffected', 'affected1', 'affected2', 'affected12']):
    """
    Infer the order of hypotheses for 2 conditions and 4 channels: ctrls, cases1, cases2, cases_both
    """
    hypotheses = {}

    mapping= {
        'unaffected': 'P(~D|V,H)',
        'affected1': 'P(D1|V,H)',
        'affected2': 'P(D2|V,H)',
        'affected12': 'P(D12|V,H)'
    }

    prevalences = (data/data.sum(1).reshape(data.shape[0], 1)).mean(0)

    probs_mean = probs.mean(0)
    probs_mean_df = pd.DataFrame(probs_mean, dtype='float32', columns=[mapping[x] for x in data_columns])
    
    pdiff = np.abs(probs_mean - prevalences)
    
    h0 = -1
    min_sum = 1
    idx = -1
    for row in pdiff:
        idx += 1
        s = row.sum()

        if s < min_sum:
            h0 = idx
            min_sum = s

    hypotheses[h0] = 'H0'
    
    h1 = (probs_mean_df['P(D1|V,H)'] - probs_mean_df['P(D2|V,H)']).idxmax()
    hypotheses[h1] = 'H1'

    h2 = (probs_mean_df['P(D2|V,H)'] - probs_mean_df['P(D1|V,H)']).idxmax()
    hypotheses[h2] = 'H2'
    
    h12 = (probs_mean_df['P(D12|V,H)']).idxmax()
    hypotheses[h12] = 'H12'

    indices = [h0, h1, h2, h12]

    l = []
    all_others = []
    for i in probs_mean_df.index:
        if i in hypotheses:
            l.append(hypotheses[i])
        else:
            all_others.append(i)

    print('probs_mean_df.index')

    probs_mean_df.index = l + all_others
    
    return np.array([h0, h1, h2, h12, *all_others]), probs_mean_df


# ordered_probs = best_dirichlet_concentrations_exp2

def get_rho_bivariate(probs, weights, concentrations, observations: np.array, n_samples: Union[int, np.array], prevalences = None, hypothesis_order = None, true_architectures: np.ndarray = None):
    """
    Requires index 3 to be cases for both and index 1 & 2 to be affected by 1, affected by 2, or vice versa
    """
    if hypothesis_order is None:
        hypothesis_order, _ = get_assumed_order_bivariate(probs, data=observations)

    ordered_weights = weights.mean(0)[hypothesis_order]
    ordered_probs = probs.mean(0)[hypothesis_order]
    ordered_concentrations = concentrations.mean(0)[hypothesis_order]
    print('ordered_weights', ordered_weights)
    print('ordered_probs', ordered_probs)

    dm_model = DirichletMultinomial(concentration=ordered_concentrations)
    # TODO: should we just take this directly from n_samples (make that a vec of ctrl, case1, case2, caseBoth)
    # TODO: name this in-sample prevalences to distinguish from population prevalence?
    # TODO: should we use population prevalence of sample prevalence below?
    if prevalences is None:
        # Alternative estimate, using weighted posterior: (ordered_weights @ ordered_probs)
        prevalences = (observations/observations.sum(1).reshape(observations.shape[0], 1)).mean(0)
        # print("prevalence estimate", prevalences)
        prevalences2 = (ordered_weights @ ordered_probs)
        print('posterior prevalence estimate', prevalences)
        print('posterior prevalence estimate using our stuff', prevalences2)
    else:
        print("Prevalence passed: ", prevalences)
        print("would have used as posterior prevalence estimate: ", (ordered_weights @ ordered_probs))

    if isinstance(n_samples, int):
        print("Estimating sample breakdown using prevalences")
        # sample_proportions = (observations/observations.sum(1).reshape(observations.shape[0], 1)).mean(0)
        n_samples = prevalences * n_samples
    alleles = n_samples*2
    alleles_ctrls, alleles_one_only, alleles_two_only, alleles_both = alleles
    print("alleles", alleles)

    alleles_ctrl1 = alleles_ctrls + alleles_two_only
    alleles_ctrl2 = alleles_ctrls + alleles_one_only
    alleles_case1 = alleles_one_only + alleles_both
    alleles_case2 = alleles_two_only + alleles_both

    P_D1 = prevalences[1]#(n_samples[3] + n_samples[1]) / (n_samples.sum())
    P_D2 = prevalences[2]#(n_samples[3] + n_samples[2]) / (n_samples.sum())

    thresh1 = Inv_Cumulative_Normal(1 - P_D1)
    thresh2 = Inv_Cumulative_Normal(1 - P_D2)

    print('P_D1', P_D1, 'P_D2', P_D2)

    n_genes_used = 0

    if not (type(true_architectures) is np.ndarray and true_architectures.size == len(observations)):
        true_architectures = [None] * len(observations)

    how_many_wrong = 0
    how_many_right = 0
    how_many_mixed_up_one_two = 0
    how_many_mixed_up_both_and_one_or_two = 0
    how_many_called_risk_when_nonrisk = 0
    how_many_called_nonrisk_when_risk = 0
        
    tot_post_neither = 0
    tot_post_one = 0
    tot_post_two = 0
    tot_post_both = 0
    tot_post_affected = 0
    
    cov_sum = 0
    sigma1_mean_est = 0
    sigma1_squared_est = 0 
    sigma2_mean_est = 0
    sigma2_squared_est = 0 
    
    n_is_both = 0
    n_is_two = 0
    n_is_one = 0
    n_is_neither = 0

    inf_rows = 0
    print_out_architecture = []
    for i, obs in enumerate(observations):
        x_ctrl, x_one, x_two, x_both = obs
        
        L_D_given_z = (ordered_probs ** obs).prod(1)
        P_z_i2 = np.multiply(ordered_weights, L_D_given_z)        
        P_z_i2 = P_z_i2/P_z_i2.sum()

        the_architecture = np.argmax(P_z_i2)
        if the_architecture == 1:
            n_is_two += 1
        elif the_architecture == 2:
            n_is_one += 1
        elif the_architecture == 0:
            n_is_neither += 1
        elif the_architecture == 3:
            n_is_both += 1

        if the_architecture == true_architectures[i]:
            how_many_right += 1
        else:
            how_many_wrong += 1

            if (the_architecture == 1 or the_architecture == 2):
                if true_architectures[i] == 1 or true_architectures[i] == 2:
                    how_many_mixed_up_one_two += 1
                elif true_architectures[i] == 3:
                    how_many_mixed_up_both_and_one_or_two += 1
                else:
                    how_many_called_risk_when_nonrisk += 1
            elif the_architecture == 3:
                if true_architectures[i] == 1 or true_architectures[i] == 2:
                    how_many_mixed_up_both_and_one_or_two += 1
                else:
                    how_many_called_risk_when_nonrisk += 1

            elif the_architecture == 0:
                how_many_called_nonrisk_when_risk += 1

        print_out_architecture.append([i, the_architecture, true_architectures[i], obs, P_z_i2])

        post_neither, post_one, post_two, post_both, *outliers = P_z_i2
        # print('outliers', outliers)

        # if the_architecture == 0:
        #     continue

        if post_both < .005:
            continue

        n_genes_used += 1

        P_V_Ctrl1 = (x_ctrl + x_two) / alleles_ctrl1
        P_V_Ctrl2 = (x_ctrl + x_one) / alleles_ctrl2
        P_V_Case1 = (x_one + x_both) / alleles_case1
        P_V_Case2 = (x_two + x_both) / alleles_case2

        # P_V_Ctrl1 = 1 - x_on1 + x_both
        P_V_1  = P_V_Ctrl1 * (1 - P_D1) + P_V_Case1 * P_D1
        P_D1_V = P_V_Case1 * P_D1 / P_V_1
        sigma1 = thresh1 - Inv_Cumulative_Normal(1 - P_D1_V)

        P_V_2  = P_V_Ctrl2 * (1 - P_D2) + P_V_Case2 * P_D2
        P_D2_V = P_V_Case2 * P_D2 / P_V_2      
        sigma2 = thresh2 - Inv_Cumulative_Normal(1 - P_D2_V)
        if np.isinf(sigma2) or np.isinf(sigma1):
            inf_rows += 1
            continue
        # print("arch vs PD", the_architecture, P_D1_V, P_D2_V)

        # print("P_D2_V", P_D2_V, "P_V_2", P_V_2, "P_D1_V", P_D1_V, "P_V_1", P_V_1)

        tot_post_neither += post_neither
        tot_post_one += post_one + post_both
        tot_post_two += post_two + post_both
        tot_post_both += post_both
        tot_post_affected += post_one + post_two + post_both

        # print("posts", post_one, post_two, post_both)
    
        sigma1_mean_est += (post_one + post_both) * sigma1
        sigma1_squared_est += (post_one + post_both) * sigma1 * sigma1
        
        sigma2_mean_est += (post_two + post_both) * sigma2
        sigma2_squared_est += (post_two + post_both) * sigma2 * sigma2
        
        cov_sum += post_both * sigma1 * sigma2
     
        i += 1

    sigma1_mean_est_f = sigma1_mean_est / tot_post_one
    sigma1_squared_est_f = sigma1_squared_est / tot_post_one
    var1_est = (sigma1_squared_est_f - sigma1_mean_est_f**2) 

    sigma2_mean_est_f = sigma2_mean_est / tot_post_two
    sigma2_squared_est_f = sigma2_squared_est / tot_post_two
    var2_est = (sigma2_squared_est_f - sigma2_mean_est_f**2) 

    cov_sum_f = cov_sum / tot_post_both
    cov = cov_sum_f - sigma1_mean_est_f*sigma2_mean_est_f

    rho = cov / (var1_est * var2_est)**.5

    print('n_genes_used', n_genes_used)
    print('tot_post_one', tot_post_one)
    print('tot_post_two', tot_post_two)
    print('tot_post_both', tot_post_both)
    print('inf_rows', inf_rows)
    print('cov_sum_f', cov_sum_f)
    print('sigma1_mean', sigma1_mean_est_f)
    print('sigma1_^2_mean', sigma1_squared_est_f)
    print('sigma2_mean', sigma2_mean_est_f)
    print('sigma2_^2_mean', sigma2_squared_est_f)
    print('cov', cov)
    print('rho', rho)

    print("how_many_wrong", how_many_wrong)
    print("how_many_right", how_many_right)
    print("how_many_mixed_up_one_two", how_many_mixed_up_one_two)
    print("how_many_mixed_up_both_and_one_or_two", how_many_mixed_up_both_and_one_or_two)
    print("how_many_called_risk_when_nonrisk", how_many_called_risk_when_nonrisk)
    print("how_many_called_nonrisk_when_risk", how_many_called_nonrisk_when_risk)

    return var1_est, var2_est, cov, rho, pd.DataFrame(print_out_architecture, columns=["i", "estimated genetic architecture", "real", "obs", 'P_z_i2'])