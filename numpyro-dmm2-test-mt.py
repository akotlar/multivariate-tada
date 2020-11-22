import numpyro
from numpyro.distributions import Multinomial, Beta, Dirichlet, DirichletMultinomial, Gamma, Beta, Categorical, Uniform, MultivariateNormal, Normal, LogNormal, Exponential, HalfCauchy, LKJCholesky
from jax import random
from numpyro.infer import MCMC, NUTS
import numpy as np
import jax.numpy as jnp
import jax
from mvl import genData
from torch import tensor

numpyro.set_host_device_count(6)
numpyro.enable_x64()

liabParams55cov = genData.genParams(pis=tensor([.1, .1, .05]), rrMeans=tensor([3., 2.]), afMean = tensor(1e-4), pDs = tensor([.01, .01]), afShape=tensor(50.), nCases=tensor([1.5e4, 1.5e4, 4e3]), nCtrls=tensor(5e4), covShared=tensor([ [1, .95], [.95, 1] ]), meanEffectCovarianceScale=tensor(.01))[0]
liabParams55cov["pDs"] = liabParams55cov["pDs"][0:2]
liabData55cov = genData.v6liability(**liabParams55cov)
liabParams55cov["pDs"] = liabData55cov["PDs"] #TODO: normalized names, and indicate that the params pDs are incomplete

def mix_weights(beta: jnp.array):
    beta_cumprod = (1 - beta).cumprod(-1)
    return jnp.pad(beta, (0,1), constant_values=1) * jnp.pad(beta_cumprod, (1,0), constant_values = 1)


# mu_exp, var_exp = get_log_params(liabParams55cov["afMean"].numpy(), 1)
# Expected number of components
# For 2 case types it's
# none, 1only, 2only, both
# For 3 it's
# none, 1only, 2only, 3only, 1&2, 1&3, 2&3, 123 (7)
# for 4 it's
# none, 1only, 2only, 3only, 4only, 1&2, 1&3, 1&4, 2&3, 2&4, 3&4, 123, 124, 134, 234, 1234
# which is 4 + 4choose2 + nchoose3  + nchoose4 
nHypotheses = 6
kConditions = 4
altCounts = liabData55cov["altCounts"].numpy()
N = len(liabData55cov["altCounts"])

nCases = liabParams55cov["nCases"].numpy()
nCtrls = liabParams55cov["nCtrls"].numpy()
empiricalAfs = altCounts.sum(1) / (nCases.sum() + nCtrls)

samplePDs = nCases / (nCases.sum() + nCtrls)
pdsAll = np.array([1 - samplePDs.sum(), *samplePDs])

# TODO: do this in numpy natively
pdsAllShaped = jnp.asarray(tensor(pdsAll).expand(nHypotheses, kConditions).numpy())
pdsAllnp = [1 - samplePDs.sum(), *samplePDs]
pdsAll = jnp.asarray([1 - samplePDs.sum(), *samplePDs])

alpha = .01

exponential_prior = jnp.array([.01]).repeat(kConditions)
def model(data):
    # This also works, for a single set of shraed parameters
    # with numpyro.plate("conc_plate", 1):
    #     conc = numpyro.sample('conc', Exponential(.001))

    with numpyro.plate("beta_plate", nHypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("prob_plate", nHypotheses):
        conc = numpyro.sample('conc', Exponential(exponential_prior).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(conc)) * pdsAll

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

def modelGamma(data):
    # This also works, for a single set of shraed parameters
    # with numpyro.plate("conc_plate", 1):
    #     conc = numpyro.sample('conc', Exponential(.001))

    with numpyro.plate("beta_plate", nHypotheses-1):
        beta = numpyro.sample("beta", Beta(1, alpha))

    with numpyro.plate("prob_plate", nHypotheses):
        conc = numpyro.sample('conc', Gamma(np.ones(pdsAllShaped.shape) / nHypotheses, 1).to_event(1))
        probs = numpyro.sample("probs", Dirichlet(conc)) * pdsAllShaped

    with numpyro.plate("data", data.shape[0]):
        z = numpyro.sample("z", Categorical(mix_weights(beta)))
        return numpyro.sample("obs", Multinomial(probs=probs[z]), obs=data)

init_rng_key = random.PRNGKey(12273)
mcmcGamma = MCMC(NUTS(modelGamma), 100, 500, jit_model_args=True, num_chains=6)
mcmcGamma.run(init_rng_key, altCounts)
mcmcGamma.print_summary()

# Scaling by pDs gives absolutely no difference in concentrations or probabilities.
concGamma = mcmcGamma.get_samples()["conc"]
print("concGamma.mean(0)\n", concGamma.mean(0))
print("\n\nconcGamma.std(0)\n", concGamma.std(0))

probsGamma = mcmcGamma.get_samples()["probs"]
print("\n\nprobsGamma.mean(0)\n",probsGamma.mean(0))
print("\n\nprobsGamma.std(0)\n",probsGamma.std(0))

# concs = mcmcAf.get_samples()["conc"]
# print("conc std", concs.std(0))
# print("conc mean", concs.mean(0))
# print("DIrichlet version", Dirichlet(concs.mean(0)).mean)

betaGamma = mcmcGamma.get_samples()['beta']
print("\ninferred stick-breaking weights", mix_weights(betaGamma).mean(0))




# pdv tensor([0.5059, 0.1545, 0.3103, 0.0828], dtype=torch.float64)
# pdv scaled tensor([0.4803, 0.1466, 0.2946, 0.0786], dtype=torch.float64)