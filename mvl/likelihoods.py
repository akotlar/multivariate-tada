
# Likelihood functions
# These assume univariate currently
from torch.multiprocessing import Process, Pool, Queue, Manager, cpu_count
import torch
import torch.tensor as tensor
# from torch.distributions import Binomial, Gamma, Uniform
from pyro.distributions import Binomial, Bernoulli, Categorical, Dirichlet, DirichletMultinomial, Beta, BetaBinomial, Uniform, Gamma, Multinomial, Gamma

import numpy as np

import scipy
from skopt import gp_minimize
from scipy.stats import binom as ScipyBinom
from matplotlib import pyplot

from collections import namedtuple
import time
# print("shape",
# TODO:
# 1) Explore constraining alphas using prevalence estimate, namely E(P(D)) = alpha0 / (alpha0 + alpha1 + alpha2 + alphaBoth) (as long as all case counts are mutually exclusive)
# 2) Can DM approximate NB + Multinomial? If so do we need mixture at all? But if we don't have that how do we model % disease-afffecting genes in each hypothesis(maybe proportion of alphas?)
# rr: relative risk
# P(V|D) = P(D|V)*P(V) / P(D)
# rr * P(D|!V) = P(D|V)
# P(V|D) = rr * P(D|!V) * P(V) / P(D)
# P(D) = (P(D|V)P(V) + P(D|!V)P(!V))
# P(D) = P(D|V) + P(D|!V)(1-P(V))
# P(V|D) = ( rrP(D|!V)) ) * P(V) ) / ( (P(D|V)P(V) + P(D|!V)(1-P(V))) )
# let a = ( rrP(D|!V)) ) * P(V) )
# P(V|D) = a / P(D|!V) / ( P(D|V)P(V) + P(D|!V) - P(D|!V)P(V) ) / P(D|!V)
# = ( rr*P(V) ) / ( rr*P(V) + 1 - P(V) )


def pVgivenD(rr, pV):
    return (rr * pV) / (rr * pV + (1 - pV))


def pVgivenDapprox(rr, pV):
    return (rr * pV)

# pD: prevalence, tensor of mConditions x 1
# pVgivenD: tensor of mConditions x 1
# pV: allele frequency


def pVgivenNotD(pD, pV, pVgivenD):
    p = (pV - (pD*pVgivenD).sum()) / (1 - pD.sum())
    if(p < 0):
        raise ValueError(
            f"pVgivenNotD: invalid params: pD: {pD}, pV: {pV}, pVgivenD: {pVgivenD} yield: p = {p}")
    return p

# def pVgivenNotD(pD, pV, pVgivenD):
#     p = (pV - (pD*pVgivenD)) / (1 - pD)
#     assert(p >= 0)
#     return p


def pDgivenV(pD, pVgivenD, pV):
    return pVgivenD * pD / pV

# pDs[0] is P(!D), prob control


def nullLikelihood(pDs, altCounts):
    return torch.exp(Multinomial(probs=pDs).log_prob(altCounts))


def effectLikelihood(nHypotheses, pDs, altCountsFlat):
    nGenes = altCountsFlat.shape[0]
    nConditions = altCountsFlat.shape[1]

    pd1 = pDs[0]
    pd2 = pDs[1]
    pdBoth = pDs[2]
    pdCtrl = 1 - pDs.sum()

    pDsAll = tensor([pdCtrl, pd1, pd2, pdBoth], dtype=torch.float64)
    
    print("pdCtrl, pd1, pd2, pdBoth: ", pDsAll)

    # nGenes x 4
    xCtrl = altCountsFlat[:, 0]
    xCase1 = altCountsFlat[:, 1]
    xCase2 = altCountsFlat[:, 2]
    xCase12 = altCountsFlat[:, 3]
    # nGenes x 1
    n = xCtrl + xCase1 + xCase2 + xCase12

    nHypothesesNonNull = nHypotheses - 1
    altCountsShaped = altCountsFlat.expand(nHypothesesNonNull, nGenes, nConditions).transpose(0, 1)
    nShaped = n.expand(nHypothesesNonNull, nGenes).T
    pdsAllShaped = pDsAll.expand(nHypothesesNonNull, nConditions)

    nullLike = nullLikelihood(pDsAll, altCountsFlat)

    def likelihoodFn(alpha0, alpha1, alpha2, alphaBoth):
        concentrations = pdsAllShaped * tensor([
            [alpha0, alpha1, alpha0, alpha1],
            [alpha0, alpha0, alpha2, alpha2],
            [alpha0, alpha1 + alphaBoth, alpha2 + alphaBoth, alpha1 + alpha2 + alphaBoth]
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

        return torch.exp(DirichletMultinomial(total_count=nShaped, concentration=concentrations).log_prob(altCountsShaped))

    return likelihoodFn, nullLike
    

def likelihoodBivariateFast(altCountsFlat, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs):
    print(altCountsFlat.shape)

    # TODO: make this flexible for multivariate
    nHypotheses = 4
    likelihoodFn, allNull2 = effectLikelihood(nHypotheses, pDs, altCountsFlat)
    def jointLikelihood(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")

        pi0 = 1.0 - (pi1 + pi2 + piBoth)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2, piBoth])
        trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])
        
        hs = tensor([[pi1, pi2, piBoth]]) * likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll
    return jointLikelihood


def writer(i, q, results):
    message = f"I am Process {i}"

    m = q.get()
    print("Result: ", m)
    return m


def processor(i, *args, **kwargs):
    np.random.seed()
    torch.manual_seed(np.random.randint(1e9))
    r = fitFnBivariate(*args, **kwargs)
    return r


def fitFnBivariateMT(altCountsFlat, pDs, nEpochs=20, minLLThresholdCount=100, K=4, debug=False, costFnIdx=0, method="nelder-mead"):
    args = [altCountsFlat, pDs, 1, minLLThresholdCount,
            K, debug, costFnIdx, method]

    results = []

    with Pool(cpu_count()) as p:
        processors = []
        for i in range(nEpochs):
            processors.append(p.apply_async(
                processor, (i, *args), callback=lambda res: results.append(res)))
        # Wait for the asynchrounous reader threads to finish
        [r.get() for r in processors]
        print("Got results")

        print(results)
        return results

# TODO: maybe beta distribution should be constrained such that variance is that of the data?
# or maybe there's an analog to 0 mean liability variance


def fitFnBivariate(altCountsByGene, pDs, nEpochs=20, minLLThresholdCount=100, K=4, debug=False, costFnIdx=0, method="nelder-mead"):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    costFn = likelihoodBivariateFast(altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)
    print("method", method, "costFn", costFn)

    assert(method == "nelder-mead" or method ==
           "annealing" or method == "basinhopping")

    llsAll = []
    lls = []
    params = []

    minLLDiff = 1
    thresholdHitCount = 0

    nGenes = len(altCountsByGene)

    # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
    # P(V|D) * P(D) / P(V)
    pi0Dist = Uniform(.5, 1)
    alphasDist = Uniform(100, 25000)

    for i in range(nEpochs):
        start = time.time()

        if method == "nelder-mead" or method == "basinhopping":
            best = float("inf")
            bestParams = []
            for y in range(200):
                pi0 = pi0Dist.sample()
                pis = Uniform(1/nGenes, 1-pi0).sample([K-1])
                pis = pis/(pis.sum() + pi0)
                fnArgs = [*pis.numpy(), *alphasDist.sample([K, ]).numpy()]

                ll = costFn(fnArgs)
                if ll < best:
                    best = ll
                    bestParams = fnArgs

            print(f"best ll: {best}, bestParams: {bestParams}")

            if method == "nelder-mead":
                fit = scipy.optimize.minimize(
                    costFn, x0=bestParams, method='Nelder-Mead', options={"maxiter": 20000, "adaptive": True})
            elif method == "basinhopping":
                fit = scipy.optimize.basinhopping(costFn, x0=bestParams)
            else:
                raise Exception("should have been nelder-mead or basinhopping")
        elif method == "annealing":
            fit = scipy.optimize.dual_annealing(costFn, [(
                0.001, .999), (.001, .999), (.001, .999), (100, 25_000), (100, 25_000), (100, 25_000), (100, 25_000)])

        print("Epoch took", time.time() - start)

        if debug:
            print(f"epoch {i}")
            print(fit)

        if not fit["success"] is True:
            if debug:
                print("Failed to converge")
                print(fit)
            continue

        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = fit["x"]
        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi1 > 1 or pi2 < 0 or pi2 > 1 or piBoth < 0 or piBoth > 1:
            if debug:
                print("Failed to converge")
                print(fit)
            continue

        ll = fit["fun"]
        llsAll.append(ll)
        if len(lls) == 0:
            lls.append(ll)
            params.append(fit["x"])
            continue

        minPrevious = min(lls)

        if debug:
            print("minPrevious", minPrevious)

        # TODO: take mode of some pc-based cluster of parameters, or some auto-encoded cluster
        if ll < minPrevious and (minPrevious - ll) >= minLLDiff:
            if debug:
                print(f"better by at >= {minLLDiff}; new ll: {fit}")

            lls.append(ll)
            params.append(fit["x"])

            thresholdHitCount = 0
            continue

        thresholdHitCount += 1

        if thresholdHitCount == minLLThresholdCount:
            break

    return {"lls": lls, "llsAll": llsAll, "params": params, "trajectoryLLs": trajectoryLLs, "trajectoryPi": trajectoryPis, "trajectoryAlphas": trajectoryAlphas}


def initBetaParams(mu, variance):
    alpha = ((1 - mu) / variance - 1 / variance) * mu**2
    beta = alpha * (1/mu - 1)

    return alpha, beta
