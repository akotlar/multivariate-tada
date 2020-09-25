# import pyro
import copy
import torch
from torch import Tensor
import torch.tensor as tensor
# import pyro.distributions as dist
from torch.multiprocessing import Process, Pool, Queue, Manager, cpu_count

from torch.distributions import Binomial, Gamma, Uniform, Categorical
from pyro.distributions import Multinomial, Dirichlet

import numpy as np

import scipy
from skopt import gp_minimize
from scipy.stats import binom as ScipyBinom
from matplotlib import pyplot

from collections import namedtuple

from .likelihoods import pVgivenD, pVgivenDapprox, pVgivenNotD, fitFnBivariate, fitFnBivariateA0, fitFnBivariateMT, fitFnBivariateStacked, fitFnBivariateStackedDirichlet, inferPDGivenVfromAlphas, empiricalPDGivenV, pVgivenNotDfromPDV
from pyper import *

import time
seed = 0

def genAlleleCount(nCtrls, nCases: Tensor, rrs = tensor([1.,1.,1.]), afMean = 1e-4, pDs = tensor([.01,.01,.002]), pNotD = None, sampleShape = (), approx = True, debug = False):
    totalSamples = nCtrls + nCases.sum()
    totalGenes = 2 * totalSamples

    # rrs: for genes affecting both D1 & D2, rr = rr1 + rr2 + rrShared
    # and then P(V|DBoth) = P(V)(rr1 + rr2 + rrShared)
    # and then P(DBoth|V) = P(V)(rr1 + rr2 + rrShared) * P(DBoth)
    # P(D1|V) = P(V)rr1  * P(D1)
    # P(D2|V) ...
    # P(V|D)
    probVgivenDs = pVgivenDapprox(rrs, afMean)
    # print("probVgivenDs", probVgivenDs)
    probVgivenNotD = pVgivenNotD(pDs, afMean, probVgivenDs)
    # print("probVgivenNotD", probVgivenNotD)
    # Q1: Is adding P(D|V) (where P(V) is fixed) ok; is adding risks ok.
    #  - log space; lognormal; Correlated topic model (a followup to DirichletMultinomials)
    # P(D|V)P(V)
    # Where P(V) is alternate af for mutational class c (PTV)
    
    # Take logs of risk and add them together
    # unusual to work on the probaiblities directly
    # P(D)*P(V|D) = P(D|V)P(V)
    # print("pDs", pDs, "1-pDs.sum()", 1 - pDs.sum())
    if pNotD is None:
        pNotD = 1-pDs.sum()
    
    totalProbabilityPopulation = tensor([probVgivenNotD*(1-pDs.sum()), *(probVgivenDs*pDs)]).sum()
    assert (abs(totalProbability-afMean) / afMean) <= 1e-6

    N = nCases.sum() + nCtrls

    PDhat = nCases / (nCases.sum() + nCtrls)
    PnotDhat = nCtrls / nCases.sum()
    # but here we need the sample 
    p = tensor([probVgivenNotD*PnotDhat, *(probVgivenDs*PDhat)])
    # print("probVgivenDs*pDs", probVgivenDs*pDs)
    # print("p", p)
    if debug:
        print(f"rrs: {rrs}, afMean: {afMean}, probVgivenNotD: {probVgivenNotD} probVgivenDs: {probVgivenDs} p: {p}")

    totalProbability = p.sum()
    # print("approx", approx)
    # print(totalProbability)
    # print(abs(totalProbability-af) / af)
    assert (abs(totalProbability-afMean) / afMean) <= 1e-6
    marginalAlleleCount = int(totalProbability * totalSamples)

    return Multinomial(probs=p, total_count=marginalAlleleCount).sample(sampleShape), p

# I used to start from P(V) and P(D). I'd calculate P(V|D), multiply that by P(D), sum and subtract from P(V)
# We can observe that P(D|V)P(V) = P(V|D)P(D)
# the sum of these P(V|D)P(D) should equal P(V)
# so P(V) - Sum_P(V|D)P(D) is the P(V|!D)P(!D)
# therefore the sum of P(D|V)P(V) that should equal P(V)
# I start from P(D|V)
# P(D|V)P(V) = P(V|D)P(D)
def genAlleleCountFromPDVS(nCases: Tensor, nCtrls: Tensor, PDVs = tensor([1.,1.,1.]), afMean = 1e-4, pDs = tensor([.01,.01,.002]), **kwargs):
    # rrs: for genes affecting both D1 & D2, rr = rr1 + rr2 + rrShared
    # and then P(V|DBoth) = P(V)(rr1 + rr2 + rrShared)
    # and then P(DBoth|V) = P(V)(rr1 + rr2 + rrShared) * P(DBoth)
    # P(D1|V) = P(V)rr1  * P(D1)
    # P(D2|V) ...
    # P(V|D)   
    assert(PDVs.shape[0] == 3)

    # One limitation is that we are constrained by the conditions we've included
    # so to get a true allele count for 
    probVgivenNotD = pVgivenNotDfromPDV(pDs, afMean, PDVs)

    # print("probVgivenNotD", probVgivenNotD)
    # print("probVgivenNotD", probVgivenNotD)
    # print("pDs", pDs, "1-pDs.sum()", 1 - pDs.sum())
    N = nCases.sum() + nCtrls
    PDhat = nCases / N
    PnotDhat = nCtrls / N

    PnotDgivenV = probVgivenNotD * PnotDhat / (1 - pDs.sum())
    PDVPV = PDVs * afMean * PDhat / pDs
    # print("PDVPV", PDVPV, "PnotD", PnotDgivenV, "probVgivenNotD", probVgivenNotD, "probVgivenNotD * PnotDhat", probVgivenNotD * PnotDhat)
    # Q1: Is adding P(D|V) (where P(V) is fixed) ok; is adding risks ok.
    #  - log space; lognormal; Correlated topic model (a followup to DirichletMultinomials)
    # P(D|V)P(V)
    # Where P(V) is alternate af for mutational class c (PTV)
    
    # Take logs of risk and add them together
    # unusual to work on the probaiblities directly
    # P(D)*P(V|D) = P(D|V)P(V)

    # print("pDs", pDs)
    # print("tensor([probVgivenNotD*(1-pDs.sum()), *(PDVs*afMean)])", tensor([probVgivenNotD*(1-pDs.sum()), *(PDVs*afMean)]))

    totalProbabilityPopulation = tensor([probVgivenNotD*(1-pDs.sum()), *(PDVs*afMean)]).sum()
    assert (abs(totalProbabilityPopulation-afMean) / afMean) <= 1e-6

    p = tensor([PnotDgivenV, *PDVPV])
    # print("p", p)

    marginalAlleleCount = int(p.sum() * N)

    return Multinomial(probs=p, total_count=marginalAlleleCount).sample(), p

# Like the 4b case, but multinomial
# TODO: shoudl we do int() or some rounding function to go from float counts to int counts
def v5(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes=20000, approx=False, **kwargs):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape,
          "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)

    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape, rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000, ]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes, ])
    rrs = rrDist.sample([nGenes, ])

    endIndices = nGenes * diseaseFractions
    startIndices = []
    for i in range(len(diseaseFractions)):
        if i == 0:
            startIndices.append(0)
            continue
        endIndices[i] += endIndices[i-1]
        startIndices.append(endIndices[i-1])

    print("startIndices", startIndices, "endIndices", endIndices)

    affectedGenes = [[]]
    unaffectedGenes = []
    rrAll = []

    totalSamples = int(nCtrls + nCases.sum())
    print("totalSamples", totalSamples)

    for geneIdx in range(nGenes):
        geneAltCounts = []
        geneProbs = []
        affects = 0
        rrSamples = tensor([1., 1., 1.])
        # Each gene gets only 1 state: affects condition 1 only, condition 2 only, or both
        # currently, in the both case, the increased in counts (rr) is. the same for both conditions
        for conditionIdx in range(nConditions):
            if geneIdx >= startIndices[conditionIdx] and geneIdx < endIndices[conditionIdx]:
                if conditionIdx == 0:
                    affects = 1
                elif conditionIdx == 1:
                    affects = 2
                elif conditionIdx == 2:
                    affects = 3
                else:
                    assert(conditionIdx <= 2)

                if len(affectedGenes) <= conditionIdx:
                    affectedGenes.append([])
                affectedGenes[conditionIdx].append(geneIdx)
                break

        assert(affects <= 3)
        # gene affects one of 3 states
        # based on which state it affects, sampleCase1, samplesCase2, samplesBoth get different rrs for this gene
        # controls always get the same value, and that is based on 1 - sum(rrs)
        if affects == 0:
            unaffectedGenes.append(geneIdx)
        elif affects == 1:
            #             print(f"affects1: {geneIdx}")
            rrSamples[0] = rrs[geneIdx, 0]
            rrSamples[2] = rrs[geneIdx, 0]  # both always gets a rr of non-1
        elif affects == 2:
            #             print(f"affects2: {geneIdx}")
            rrSamples[1] = rrs[geneIdx, 1]
            rrSamples[2] = rrs[geneIdx, 1]
        elif affects == 3:
            #             print(f"affects2: {geneIdx}")
            rrSamples[0] = rrs[geneIdx, 0] + rrs[geneIdx, 2]
            rrSamples[1] = rrs[geneIdx, 1] + rrs[geneIdx, 2]
            rrSamples[2] = rrs[geneIdx, 0] + rrs[geneIdx, 1] + rrs[geneIdx, 2]

        altCountsGene, p = genAlleleCount(nCases = nCases, nCtrls = nCtrls, rrs = rrSamples, afMean = afs[geneIdx], pDs = pDs,approx=approx)

        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts, dtype=torch.short)
    probs = tensor(probs, dtype=torch.float64)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "affectedGenes": tensor(affectedGenes, dtype=torch.short), "unaffectedGenes": tensor(unaffectedGenes, dtype=torch.short), "rrs": rrAll}

# Like 5, but make approximation that P(V|D) = P(V)*rr, by observing that rr*P(D|V) + 1-P(V) is ~1 for intermediate rr and small P(V)
# say a typical P(V|D) is ~
def v6(nCases, nCtrls, pDs = tensor([.01, .01, .002]), diseaseFractions = tensor([.05, .05, .05]), rrShape = tensor(50.), rrMeans = tensor([3., 3., 3.]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=tensor(20_000), approx=True, rrtype="multiplicative", **kwargs):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape,
          "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs, "rrtype", rrtype)

    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape, rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000, ]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes, ])
    rrs = rrDist.sample([nGenes, ])

    endIndices = nGenes * diseaseFractions
    startIndices = []
    for i in range(len(diseaseFractions)):
        if i == 0:
            startIndices.append(0)
            continue
        endIndices[i] += endIndices[i-1]
        startIndices.append(endIndices[i-1])

    print("startIndices", startIndices, "endIndices", endIndices)

    affectedGenes = [[]]
    unaffectedGenes = []
    rrAll = []

    totalSamples = int(nCtrls + nCases.sum())
    print("totalSamples", totalSamples)
    for geneIdx in range(nGenes):
        geneAltCounts = []
        geneProbs = []
        affects = 0
        rrSamples = tensor([1., 1., 1.])
        # Each gene gets only 1 state: affects condition 1 only, condition 2 only, or both
        # currently, in the both case, the increased in counts (rr) is. the same for both conditions
        for conditionIdx in range(nConditions):
            if geneIdx >= startIndices[conditionIdx] and geneIdx < endIndices[conditionIdx]:
                if conditionIdx == 0:
                    affects = 1
                elif conditionIdx == 1:
                    affects = 2
                elif conditionIdx == 2:
                    affects = 3
                else:
                    assert(conditionIdx <= 2)

                if len(affectedGenes) <= conditionIdx:
                    affectedGenes.append([])
                affectedGenes[conditionIdx].append(geneIdx)
                break

        assert(affects <= 3)
        # gene affects one of 3 states
        # based on which state it affects, sampleCase1, samplesCase2, samplesBoth get different rrs for this gene
        # controls always get the same value, and that is based on 1 - sum(rrs)
        if affects == 0:
            unaffectedGenes.append(geneIdx)
        elif affects == 1:
            #             print(f"affects1: {geneIdx}")
            rrSamples[0] = rrs[geneIdx, 0]
            rrSamples[2] = rrSamples[0]
        elif affects == 2:
            #             print(f"affects2: {geneIdx}")
            rrSamples[1] = rrs[geneIdx, 1]
            rrSamples[2] = rrSamples[1]
        elif affects == 3:
            if rrtype == "multiplicative":
                rrSamples[0] = rrs[geneIdx, 0] * rrs[geneIdx, 2]
                rrSamples[1] = rrs[geneIdx, 1] * rrs[geneIdx, 2]
                rrSamples[2] = rrs[geneIdx, 0] * rrs[geneIdx, 1] * rrs[geneIdx, 2]
            elif rrtype == "unique":
                rrSamples[0] = rrs[geneIdx, 0]
                rrSamples[1] = rrs[geneIdx, 1]
                rrSamples[2] = rrs[geneIdx, 2]
            else:
                rrSamples[0] = rrs[geneIdx, 0] + rrs[geneIdx, 2]
                rrSamples[1] = rrs[geneIdx, 1] + rrs[geneIdx, 2]
                rrSamples[2] = rrs[geneIdx, 0] + rrs[geneIdx, 1] + rrs[geneIdx, 2]

        altCountsGene, p = genAlleleCount(nCases = nCases, nCtrls = nCtrls, rrs = rrSamples, afMean = afs[geneIdx], pDs = pDs, approx=approx)

        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll}

# v6 with 3 components
def v6_3(nCases, nCtrls, pDs = tensor([.01, .01, .01, .002, .002, .002, .002]), diseaseFractions = tensor([.05, .05, .05, .05, .05, .05]), rrShape = tensor(50.), rrMeans = tensor([3., 3., 3., 1.5, 1.5, 1.5, 1.5]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=tensor(20_000), approx=True, **kwargs):
    # TODO: build into matrix
    lookup = {
        "1": 0, "2": 1, "3": 2, "12": 3, "13": 4, "23": 5, "123": 6
    }

    nSampleTypes = len(lookup)
    assert len(pDs) == nSampleTypes and len(diseaseFractions) == nSampleTypes and len(rrMeans) == nSampleTypes

    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape,
          "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)

    nConditions = len(nCases)
    assert(nConditions == nSampleTypes)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape, rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000, ]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes, ])
    rrs = rrDist.sample([nGenes, ])

    endIndices = nGenes * diseaseFractions
    startIndices = []
    for i in range(len(diseaseFractions)):
        if i == 0:
            startIndices.append(0)
            continue
        endIndices[i] += endIndices[i-1]
        startIndices.append(endIndices[i-1])

    print("startIndices", startIndices, "endIndices", endIndices)

    affectedGenes = [[]]
    unaffectedGenes = []
    rrAll = []

    totalSamples = int(nCtrls + nCases.sum())
    print("totalSamples", totalSamples)
    for geneIdx in range(nGenes):
        geneAltCounts = []
        geneProbs = []
        affects = -1
        rrSamples = tensor([1.]).repeat(nConditions)
        # Each gene gets only 1 state: affects condition 1 only, condition 2 only, or both
        # currently, in the both case, the increased in counts (rr) is. the same for both conditions
        for conditionIdx in range(nConditions):
            if geneIdx >= startIndices[conditionIdx] and geneIdx < endIndices[conditionIdx]:
                affects = conditionIdx

                if len(affectedGenes) <= conditionIdx:
                    affectedGenes.append([])
                affectedGenes[conditionIdx].append(geneIdx)
                break

        # gene affects one of 3 states
        # based on which state it affects, sampleCase1, samplesCase2, samplesBoth get different rrs for this gene
        # controls always get the same value, and that is based on 1 - sum(rrs)
        if affects == -1:
            unaffectedGenes.append(geneIdx)
        elif affects == lookup["1"]:
            effect = rrs[geneIdx, 0]
            rrSamples[lookup["1"]] = effect
            rrSamples[lookup["12"]] = effect
            rrSamples[lookup["13"]] = effect
            rrSamples[lookup["123"]] = effect
        elif affects == lookup["2"]:
            effect = rrs[geneIdx, 1]
            rrSamples[lookup["2"]] = effect
            rrSamples[lookup["12"]] = effect
            rrSamples[lookup["23"]] = effect
            rrSamples[lookup["123"]] = effect
        elif affects == lookup["3"]:
            effect = rrs[geneIdx, 2]
            rrSamples[lookup["3"]] = effect
            rrSamples[lookup["13"]] = effect
            rrSamples[lookup["23"]] = effect
            rrSamples[lookup["123"]] = effect
        elif affects == lookup["12"]:
            effect1 = rrs[geneIdx, lookup["1"]]
            effect2 = rrs[geneIdx, lookup["2"]]
            effect12 = rrs[geneIdx, lookup["12"]]

            rrSamples[lookup["1"]] = effect1 + effect12
            rrSamples[lookup["13"]] = effect1 + effect12
            rrSamples[lookup["2"]] = effect2 + effect12
            rrSamples[lookup["23"]] = effect2 + effect12
            rrSamples[lookup["12"]] = effect1 + effect2 + effect12
            rrSamples[lookup["123"]] = effect1 + effect2 + effect12
        elif affects == lookup["13"]:
            effect1 = rrs[geneIdx, lookup["1"]]
            effect2 = rrs[geneIdx, lookup["3"]]
            effect12 = rrs[geneIdx, lookup["13"]]

            rrSamples[lookup["1"]] = effect1 + effect12
            rrSamples[lookup["3"]] = effect2 + effect12
            rrSamples[lookup["13"]] = effect1 + effect2 + effect12
            rrSamples[lookup["23"]] = rrSamples[lookup["2"]]
            rrSamples[lookup["123"]] = rrSamples[lookup["12"]]
        elif affects == lookup["23"]:
            effect1 = rrs[geneIdx, lookup["2"]]
            effect2 = rrs[geneIdx, lookup["3"]]
            effect12 = rrs[geneIdx, lookup["23"]]
            rrSamples[lookup["2"]] = effect1 + effect12
            rrSamples[lookup["3"]] = effect2 + effect12
            rrSamples[lookup["23"]] = effect1 + effect2 + effect12
            rrSamples[lookup["13"]] = rrSamples[lookup["2"]]
            rrSamples[lookup["123"]] = rrSamples[lookup["23"]]
        # finish
        elif affects == lookup["123"]:
            effect1 = rrs[geneIdx, lookup["1"]]
            effect2 = rrs[geneIdx, lookup["2"]]
            effect3 = rrs[geneIdx, lookup["3"]]
            effect12 = rrs[geneIdx, lookup["12"]]
            effect13 = rrs[geneIdx, lookup["13"]]
            effect23 = rrs[geneIdx, lookup["23"]]
            effect123 = rrs[geneIdx, lookup["123"]]
            rrSamples[lookup["1"]] = effect1 + effect123
            rrSamples[lookup["2"]] = effect2 + effect123
            rrSamples[lookup["3"]] = effect3 + effect123
            rrSamples[lookup["12"]] = effect1 + effect2 + effect123
            rrSamples[lookup["13"]] = effect1 + effect3 + effect123
            rrSamples[lookup["23"]] = effect2 + effect3 + effect123
            rrSamples[lookup["123"]] = effect1 + effect2 + effect3 + effect123

        altCountsGene, p = genAlleleCount(nCases = nCases, nCtrls = nCtrls, rrs = rrSamples, afMean = afs[geneIdx], pDs = pDs, approx=approx)

        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll}

# def v6(nCases, nCtrls, pDs = tensor([.01, .01, .002]), diseaseFractions = tensor([.05, .05, .05]), rrShape = tensor(50.), rrMeans = tensor([3., 3., 3.]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=tensor(20_000), approx=True, rrtype='default', **kwargs):
#     # TODO: assert shapes match
#     print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape,
#           "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs, "rrtype", rrtype)

#     nConditions = len(nCases)
#     assert(nConditions == 3)
#     altCounts = []
#     probs = []
#     afDist = Gamma(concentration=afShape, rate=afShape/afMean)
#     rrDist = Gamma(concentration=rrShape, rate=rrShape/rrMeans)
#     print("rrDist mean", rrDist.sample([10_000, ]).mean(0))
# #     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))

#     # shape == [nGenes, nConditions]
#     afs = afDist.sample([nGenes, ])
#     rrs = rrDist.sample([nGenes, ])

#     endIndices = nGenes * diseaseFractions
#     startIndices = []
#     for i in range(len(diseaseFractions)):
#         if i == 0:
#             startIndices.append(0)
#             continue
#         endIndices[i] += endIndices[i-1]
#         startIndices.append(endIndices[i-1])

#     print("startIndices", startIndices, "endIndices", endIndices)

#     affectedGenes = [[]]
#     unaffectedGenes = []
#     rrAll = []

#     totalSamples = int(nCtrls + nCases.sum())
#     print("totalSamples", totalSamples)
#     for geneIdx in range(nGenes):
#         geneAltCounts = []
#         geneProbs = []
#         affects = 0
#         rrSamples = tensor([1., 1., 1.])
#         # Each gene gets only 1 state: affects condition 1 only, condition 2 only, or both
#         # currently, in the both case, the increased in counts (rr) is. the same for both conditions
#         for conditionIdx in range(nConditions):
#             if geneIdx >= startIndices[conditionIdx] and geneIdx < endIndices[conditionIdx]:
#                 if conditionIdx == 0:
#                     affects = 1
#                 elif conditionIdx == 1:
#                     affects = 2
#                 elif conditionIdx == 2:
#                     affects = 3
#                 else:
#                     assert(conditionIdx <= 2)

#                 if len(affectedGenes) <= conditionIdx:
#                     affectedGenes.append([])
#                 affectedGenes[conditionIdx].append(geneIdx)
#                 break

#         assert(affects <= 3)
#         # gene affects one of 3 states
#         # based on which state it affects, sampleCase1, samplesCase2, samplesBoth get different rrs for this gene
#         # controls always get the same value, and that is based on 1 - sum(rrs)
#         if affects == 0:
#             unaffectedGenes.append(geneIdx)
#         elif affects == 1:
#             #             print(f"affects1: {geneIdx}")
#             rrSamples[0] = rrs[geneIdx, 0]
#             rrSamples[2] = rrSamples[0]
#         elif affects == 2:
#             #             print(f"affects2: {geneIdx}")
#             rrSamples[1] = rrs[geneIdx, 1]
#             rrSamples[2] = rrSamples[1]
#         elif affects == 3:
#             # print("affects 3")
#             #             print(f"affects2: {geneIdx}")
#             rrSamples[0] = rrs[geneIdx, 0] + rrs[geneIdx, 2]
#             rrSamples[1] = rrs[geneIdx, 1] + rrs[geneIdx, 2]
#             rrSamples[2] = rrs[geneIdx, 0] + rrs[geneIdx, 1] + rrs[geneIdx, 2]

#         altCountsGene, p = genAlleleCount(nCases = nCases, nCtrls = nCtrls, rrs = rrSamples, afMean = afs[geneIdx], pDs = pDs, approx=approx)

#         altCounts.append(altCountsGene.numpy())
#         probs.append(p.numpy())
#         rrAll.append(rrSamples)
#     altCounts = tensor(altCounts)
#     probs = tensor(probs)

#     # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
#     return {"altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll}

# Like 6 but generates correlated relative risks by sampling from lognormal
def v6normal(nCases: Tensor, nCtrls: Tensor, pDs = tensor([.01, .01, .002]), pNotD = None, diseaseFractions = tensor([.05, .05, .05]), rrMeans = tensor([3, 3,  3]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
             covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), covSingle=tensor([[1, 0], [0, 1]]), approx=True, rrtype='default', **kwargs):
    # print("old", old)
    # TODO: assert shapes match
    print("TESTING v6normal WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "afMean", afMean,
          "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs, "covShared", covShared, "covSingle", covSingle)
    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)

    r = R(use_pandas=True)
    
    if rrtype == "unique" or rrtype == "unique-multiplicative" or rrtype=='default':
        covSharedStr = "c(" + ",".join([str(x) for x in covShared.view(-1).numpy()]) + ")"
        covSingleStr = "c(" + ",".join([str(x) for x in covSingle.view(-1).numpy()]) + ")"
        # r = R(use_pandas=True)
        # r(f'''
        #     library(tmvtnorm)
        #     sigma <- matrix(c({covSharedStr}), ncol={len(covShared)})
        #     rrsShared <- rtmvnorm(n={nGenes}, mean=c({rrMeans[0] + rrMeans[2]}, {rrMeans[1] + rrMeans[2]}, {rrMeans[0] + rrMeans[1] + rrMeans[2]}), sigma=sigma, lower=c(1,1,1))
        #     sigma <- matrix(c({covSingleStr}), ncol={len(covSingle)})
        #     rrsOne <- rtmvnorm(n={nGenes}, mean=c({rrMeans[0]}, {rrMeans[1]}), sigma=sigma, lower=c(1,1))
        # ''')
        # rrsShared = tensor(r.get('rrsShared'))
        # rrsOne = tensor(r.get('rrsOne'))

        print("covSharedStr", covSharedStr)
        print("covSingleStr", covSingleStr)
        singleMeans = f"c({rrMeans[0]}, {rrMeans[1]})"

        if rrtype == "unique":
            sharedMeans = f"c({rrMeans[0]}, {rrMeans[1]}, {rrMeans[0] + rrMeans[1]})"
        elif rrtype == "unique-multiplicative":
            print("running unique multiplicative")
            sharedMeans = f"c({rrMeans[0]}, {rrMeans[1]}, {rrMeans[0] * rrMeans[1]})"
        else:
            sharedMeans = f"c({rrMeans[0] + rrMeans[2]}, {rrMeans[1] + rrMeans[2]}, {rrMeans[0] + rrMeans[1] + rrMeans[2]})"
        
        print("sharedMeans", sharedMeans)
        print("singleMeans", singleMeans)
        print("len covshared", len(covShared))
        r(f'''
            library(tmvtnorm)
            sigma <- matrix({covSharedStr}, ncol={len(covShared)})
            rrsShared <- rtmvnorm(n={nGenes}, mean={sharedMeans}, sigma=sigma, lower=c(1,1,1))
            sigma <- matrix({covSingleStr}, ncol={len(covSingle)})
            rrsOne <- rtmvnorm(n={nGenes}, mean={singleMeans}, sigma=sigma, lower=c(1,1))
        ''')
        print("r.get('rrsOne')", r.get('rrsOne'))
        rrsShared = tensor(r.get('rrsShared'))
        print("rrsShared", rrsShared)
        rrsOne = tensor(r.get('rrsOne'))
        print("rrsOne", rrsOne)
    elif rrtype == 'lognormal-unique':
        print("RUNNING NEW")
        assert len(rrMeans) == 2 and len(covShared) == 2
        muRR = np.log(rrMeans) #There is no longer a  "shared" contribution
        rrsOne = tensor(np.random.multivariate_normal(muRR, covSingle, size=nGenes))
        rrsShared = tensor(np.random.multivariate_normal(muRR, covShared, size=nGenes))
        rrSum = tensor([rrsShared.sum(1).numpy()])
        rrsShared = torch.exp(torch.cat([rrsShared, rrSum.T], 1))
        print(rrsShared)
    elif rrtype == 'libaility':
        pass
    else:
        raise Exception("not understood")
        
    print("rrsShared", rrsShared, "means", rrsShared.mean(0))
    print("rrShared correlation 1 & 2", np.corrcoef(rrsShared[:,0], rrsShared[:,1]))
    print("rrShared correlation 1 & 3", np.corrcoef(rrsShared[:,0], rrsShared[:,2]))

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes, ])

    affectedGenes = []
    unaffectedGenes = []
    rrAll = []

    totalSamples = int(nCtrls + nCases.sum())
    print("totalSamples", totalSamples)

    assert diseaseFractions.sum() <= 1
    pis = tensor([1 - diseaseFractions.sum(), *diseaseFractions])

    for i in range(len(pis) - 1):
        affectedGenes.append([])

    print("pis", pis)
    regimeSampler = Categorical(pis)
    geneArchitecture = regimeSampler.sample([nGenes,])
    print('geneArchitecture', len(torch.nonzero(geneArchitecture == 1)), len(torch.nonzero(geneArchitecture == 2)), len(torch.nonzero(geneArchitecture == 3)), )
    for geneIdx in range(nGenes):
        affects = 0
        rrSamples = tensor([1., 1., 1.])
        
        affects = geneArchitecture[geneIdx]

        if affects == 0:
            unaffectedGenes.append(geneIdx)
        else:
            affectedGenes[affects-1].append(geneIdx)
            if affects == 1:
                # TODO: do we need to have 0 correlation between rrSamples[0] and rrSampels[2]
                rrSamples[0] = rrsOne[geneIdx, 0]
                rrSamples[2] = rrSamples[0]
            elif affects == 2:
                rrSamples[1] = rrsOne[geneIdx, 1]
                rrSamples[2] = rrSamples[1]
            elif affects == 3:
                rrSamples = rrsShared[geneIdx]
        
        altCountsGene, p = genAlleleCount(nCtrls = nCtrls, nCases = nCases, rrs = rrSamples, afMean = afs[geneIdx], pDs = pDs, pNotD = pNotD, approx=approx)
       
        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll}

def makeCovarianceMatrix(corrMatrix: Tensor, variances: Tensor):
    return corrMatrx * torch.prod(variances)

# With rr == 1, P(V|D) == P(V) * rr
# RR's are calculated from mean effects; the liability shift that is evidenced by 
# the population prevalence given exposure (which is the mean effect)
# pDs here is population prevalence, and afs are population afs
def v6liability(nCases, nCtrls, pDs = tensor([.01, .01, .002]), diseaseFractions = tensor([.05, .05, .01]), rrMeans = tensor([3, 5]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
             covShared=tensor([[.1, .04], [.04, .1]]), covSingle=tensor([[.1, 0], [0, .1]]), **kwargs):
    from torch.distributions import MultivariateNormal as MVN, Categorical, Normal as N
    from torch import Tensor
    import numpy as np

    def sampleNonNegative(mean: Tensor, cov: Tensor):
        mvn = MVN(mean, cov)
        samples = mvn.sample()

        for sample in samples:
            if sample < 0:
                return sampleNonNegative(mean, cov)

        return samples

    def sampleNnonNeg(N: int, mean: Tensor, cov: Tensor):
        print("mean", mean)
        res = []
        for i in range(N):
            n = sampleNonNegative(mean, cov)
            res.append(n.numpy())

        return tensor(res)

    def getTargetMeanEffect(PD: Tensor, rrTarget: Tensor):
        norm = N(0, 1)
        pdThresh = norm.icdf(1 - PD)
        pdTarget = PD * rrTarget
        print("pdThresh", pdThresh)
        print("pdTarget", pdTarget)
        pdvthresh = norm.icdf(1 - pdTarget)
        print("pdvthresh", pdvthresh)
        meanEffect = pdThresh - pdvthresh
        print("meanEffect", meanEffect)
        return meanEffect

    # Q: do we sample from normal distribution 
    norm = N(0, 1)
    pdBoth = pDs[2]
    pdsOne = pDs[0:2]

    # print("pds", pDs, "pds", pds, "pdsBoth", pdBoth)

    targetMeanEffects = getTargetMeanEffect(pdsOne, rrMeans)
    testSampleRestricted = sampleNnonNeg(nGenes, targetMeanEffects, covShared)
    totalRestricted = testSampleRestricted[:, 0] + testSampleRestricted[:, 1]

    pdsThresh = norm.icdf(1 - pdsOne)
    pdBothThresh = norm.icdf(1 - pdBoth)
    PDOneGivenBothAffectedThresh = pdsThresh - testSampleRestricted
    PDBothGivenBothAffectedThresh = pdBothThresh - totalRestricted
    PDoneGivenV = 1 - norm.cdf(PDOneGivenBothAffectedThresh)
    PDbothGivenV = 1 - norm.cdf(PDBothGivenBothAffectedThresh)

    singleMeanEffects = sampleNnonNeg(nGenes, targetMeanEffects, covSingle)
    # nullEffects = sampleNnonNeg(1000, tensor([0., 0.]), covSingle)
    # print("mean effects means", meanEffects.mean(0))
    # print("null effects means", nullEffects.mean(0))
    # print("(meanEffects/nullEffects).mean(0)", (nullEffects).mean(0))

    PDVonlyOneThresh = pdsThresh - singleMeanEffects
    PDVBothIn1Thresh = pdBothThresh - singleMeanEffects[:, 0]
    PDVBothIn2Thresh = pdBothThresh - singleMeanEffects[:, 1]

    PDVonlyOne = 1 - norm.cdf(PDVonlyOneThresh)
    PDBothIn1 = 1 - norm.cdf(PDVBothIn1Thresh)
    PDBothIn2 = 1 - norm.cdf(PDVBothIn2Thresh)

    d0affected = pDs.expand(nGenes, 3)
    d1affected = torch.stack([PDVonlyOne[:,0], pdsOne[1].expand(PDVonlyOne.shape[0]), PDBothIn1]).T
    d2affected = torch.stack([pdsOne[0].expand(PDVonlyOne.shape[0]), PDVonlyOne[:,1], PDBothIn2]).T
    bothAffected = torch.stack([*PDoneGivenV.T, PDbothGivenV]).T
    
    #shape nGenes, kGeneticArchitectureCategories, mSampleCategories
    PDVs = torch.stack([d0affected, d1affected, d2affected, bothAffected]).transpose(1,0)

    # print("d0affected", d0affected)
    # print("d1affected", d1affected.mean(0))
    # print("d2affected", d2affected.mean(0))
    # print("bothAffected", bothAffected.mean(0))
    print("PDVs", PDVs, "PDVs.shape", PDVs.shape)

    # creates nGene * len(pDs) tensor
    # probably a  way of doing this using cat
    # PDVs = torch.stack([*PDoneGivenV.T,PDbothGivenV]).T

    # shape == [nGenes, nConditions]
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    afs = afDist.sample([nGenes, ])

    pis = tensor([1 - diseaseFractions.sum(), *diseaseFractions])
    categorySampler = Categorical(pis)
    categories = categorySampler.sample([nGenes,])

    affectedGenes = []
    for i in range(len(pis)):
        affectedGenes.append([])

    unaffectedGenes = []
    altCounts = []
    probs = []

    for geneIdx in range(nGenes):
        affects = categories[geneIdx]

        if affects == 0:
            unaffectedGenes.append(geneIdx)
        else:
            affectedGenes[affects - 1].append(geneIdx)

        altCountsGene, p = genAlleleCountFromPDVS(nCases = nCases, nCtrls = nCtrls, PDVs = PDVs[geneIdx, affects], afMean = afs[geneIdx], pDs = pDs)
       
        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())

    altCounts.append(altCountsGene.numpy())
    probs.append(p.numpy())
    altCounts = tensor(altCounts)
    probs = tensor(probs)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "categories": categories, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "PDVs": PDVs}

    # print("pdBothThresh", pdBothThresh)
    # print("PDBothGivenVthreshold", PDBothGivenVthreshold)
    # print("totalRestricted", totalRestricted)
    # print("PDbothGivenV", PDbothGivenV)
    # norm.cdf(PDBothGivenVthreshold.mean())
# Like 6, but only 2 groups of genes, those that affect 1only, or 2only. Samples that have both conditions just get rr1 in 1 genes, rr2 in 2 genes
# so the trick is that we have no 3rd component to infer; our algorithm should place minimal weight on that component
# if given 3 diseaseFractions, 3rd is ignored
def v6twoComponents(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes=20000, approx = True):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape,
          "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)

    diseaseFractions = diseaseFractions[0:-1]  # bad, no reassign recommended
    nConditions = len(nCases) - 1
    assert(nConditions == 2)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape, rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000, ]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes, ])
    rrs = rrDist.sample([nGenes, ])

    endIndices = nGenes * diseaseFractions
    startIndices = []
    for i in range(nConditions):
        if i == 0:
            startIndices.append(0)
            continue
        endIndices[i] += endIndices[i-1]
        startIndices.append(endIndices[i-1])

    print("startIndices", startIndices, "endIndices", endIndices)

    affectedGenes = [[]]
    unaffectedGenes = []
    rrAll = []

    totalSamples = int(nCtrls + nCases.sum())
    print("totalSamples", totalSamples)

    for geneIdx in range(nGenes):
        geneAltCounts = []
        geneProbs = []
        affects = 0
        # still 3, brecause we still have a "both" category
        rrSamples = tensor([1., 1., 1.])
        # Each gene gets only 1 state: affects condition 1 only, condition 2 only, or both
        # currently, in the both case, the increased in counts (rr) is. the same for both conditions
        for conditionIdx in range(nConditions):
            if geneIdx >= startIndices[conditionIdx] and geneIdx < endIndices[conditionIdx]:
                if conditionIdx == 0:
                    affects = 1
                elif conditionIdx == 1:
                    affects = 2
                else:
                    assert(conditionIdx <= 2)

                if len(affectedGenes) <= conditionIdx:
                    affectedGenes.append([])
                affectedGenes[conditionIdx].append(geneIdx)
                break

        assert(affects <= 2)
        # gene affects one of 3 states
        # based on which state it affects, sampleCase1, samplesCase2, samplesBoth get different rrs for this gene
        # controls always get the same value, and that is based on 1 - sum(rrs)
        if affects == 0:
            unaffectedGenes.append(geneIdx)
        elif affects == 1:
            #             print(f"affects1: {geneIdx}")
            rrSamples[0] = rrs[geneIdx, 0]
            rrSamples[2] = rrSamples[0]
        elif affects == 2:
            #             print(f"affects2: {geneIdx}")
            rrSamples[1] = rrs[geneIdx, 1]
            rrSamples[2] = rrSamples[1]

        altCountsGene, p = genAlleleCount(nCases = nCases, nCtrls = nCtrls, rrs = rrSamples, afMean = afs[geneIdx], pDs = pDs, approx = approx)

        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll}

# Like the 6 case, but we scale P(V|Ds) by prevalence, since the actual sample sizes say for the binomial in which P(V|D1) would be used is the fraction P(D1) of the total
# and in the multionmial setting, we use only a single sample size
# for instance, lets say we have 500k controls, 1000 cases
# the P(V|D) (cases) may be .0001 and P(V|!D) may  .0001, but the probability in a multinomial should really be 99.9999% in favor of controls
def v7(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes=20000):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape,
          "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)

    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape, rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000, ]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes, ])
    rrs = rrDist.sample([nGenes, ])

    endIndices = nGenes * diseaseFractions
    startIndices = []
    for i in range(len(diseaseFractions)):
        if i == 0:
            startIndices.append(0)
            continue
        endIndices[i] += endIndices[i-1]
        startIndices.append(endIndices[i-1])

    print("startIndices", startIndices, "endIndices", endIndices)

    affectedGenes = [[]]
    unaffectedGenes = []

    totalSamples = int(nCtrls + nCases.sum())

    print("totalSamples", totalSamples)
    for geneIdx in range(nGenes):
        geneAltCounts = []
        geneProbs = []
        affects = 0

        # Each gene gets only 1 state: affects condition 1 only, condition 2 only, or both
        # currently, in the both case, the increased in counts (rr) is. the same for both conditions
        for conditionIdx in range(nConditions):
            if geneIdx >= startIndices[conditionIdx] and geneIdx < endIndices[conditionIdx]:
                if conditionIdx == 0:
                    affects = 1
                elif conditionIdx == 1:
                    affects = 2
                elif conditionIdx == 2:
                    affects = 3
                else:
                    assert(conditionIdx <= 2)

                if len(affectedGenes) <= conditionIdx:
                    affectedGenes.append([])
                affectedGenes[conditionIdx].append(geneIdx)
                break

        assert(affects <= 3)

        PVDcases = pVgivenD(tensor([1., 1., 1.]), afs[geneIdx])
        # gene affects one of 3 states
        # based on which state it affects, sampleCase1, samplesCase2, samplesBoth get different rrs for this gene
        # controls always get the same value, and that is based on 1 - sum(rrs)
        if affects == 0:
            unaffectedGenes.append(geneIdx)
        elif affects == 1:
            PVDcases[0] = pVgivenD(rrs[geneIdx, 0], afs[geneIdx])
            PVDcases[2] = PVDcases[0]
        elif affects == 2:
            PVDcases[1] = pVgivenD(rrs[geneIdx, 1], afs[geneIdx])
            PVDcases[2] = PVDcases[0]
        elif affects == 3:
            pvds = pVgivenD(rrs[geneIdx], afs[geneIdx])
            PVDcases[0] = pvds[0] + pvds[2]
            PVDcases[1] = pvds[1] + pvds[2]
            PVDcases[2] = pvds[0] + pvds[1] + pvds[2]

        PVNotD = pVgivenNotD(pDs, afs[geneIdx], PVDcases)  # * (1 - pDs.sum())
        PVDcases = PVDcases  # * pDs

        # P(D|V)/P(V)
        PVDprevalenceWeighted = PVDcases * pDs
        PVNotDprevalenceWeighted = PVNotD * (1 - pDs.sum())
        totalProbability = PVDprevalenceWeighted.sum() + PVNotDprevalenceWeighted
#         print("affects", affects, "af", afs[geneIdx], "PVDcases", PVDcases, "pDs", pDs, "PVNotD", PVNotD, "totalProbability", totalProbability)

        assert abs(totalProbability-afs[geneIdx]) / afs[geneIdx] < 0.00001
        marginalAlleleCount = int(totalProbability * totalSamples)
#         print("marginal allele count", marginalAlleleCount)

        p = tensor([PVNotDprevalenceWeighted, *PVDprevalenceWeighted])
#         print("probs", probs)
        # without .numpy() can't later convert tensor(altCounts) : "only tensors can be converted to Python scalars"
        altCountsGene = Multinomial(
            probs=p, total_count=marginalAlleleCount).sample().numpy()

#         print("altCountsGene", altCountsGene)
        altCounts.append(altCountsGene)

        probs.append(p.numpy())
    altCounts = tensor(altCounts)
    probs = tensor(probs)

    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return {"altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": tensor(rrs)}


def flattenAltCounts(altCounts, afs):
    altCountsFlatPooled = []
    afsFlatPooled = []
    for geneIdx in range(nGenes):
        altCountsFlatPooled.append(
            [altCounts[geneIdx, 0, 0], *altCounts[geneIdx, :, 1].flatten()])
        afsFlatPooled.append(
            [afs[geneIdx, 0, 0], *afs[geneIdx, :, 1].flatten()])

    altCountsFlatPooled = tensor(altCountsFlatPooled)
    afsFlatPooled = tensor(afsFlatPooled)
    print("altCountsFlatPooled", altCountsFlatPooled)
    print("afsFlatPooled", afsFlatPooled)

    flattenedData = []

    for geneAfData in afs:
        flattenedData.append([geneAfData[0][0], *geneAfData[:, 1]])
    flattenedData = tensor(flattenedData)

    return altCountsFlatPooled, afsFlatPooled, flattenedData


def genParams(pis=tensor([.1, .1, .05]), rrShape=tensor(10.), rrMeans=tensor([3., 3., 1.5]), afShape=tensor(10.), afMean=tensor(1e-4), nCases=tensor([5e3, 5e3, 2e3]), nCtrls=tensor(5e5), covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), covSingle=tensor([[1, 0], [0, 1]]), pDs=None, rrtype="default"):
    nGenes = 20_000

    if pDs is None:
        pDs = nCases.true_divide(nCases.sum() + nCtrls)
    print("pDs are:", pDs)
    print("covShared is", covShared)
    return [{
        "nGenes": nGenes,
        "nCases": nCases,
        "nCtrls": nCtrls,
        "pDs": pDs,
        "diseaseFractions": pis,
        "rrShape": rrShape,
        "rrMeans": rrMeans,
        "afShape": afShape,
        "afMean": afMean,
        "covShared": covShared,
        "covSingle": covSingle,
        "rrtype": rrtype
    }]


def writer(i, q, results):
    message = f"I am Process {i}"

    m = q.get()
    print("Result: ", m)
    return m


def processor(i, params, kwargs):
    print("kwargs", kwargs)
    np.random.seed()
    torch.manual_seed(np.random.randint(1e9))
    r = runSimIteration(params, **kwargs)
    return r


def runSimMT(rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.05, .05, .05]]),
             nCases=tensor([15e3, 15e3, 6e3]), nCtrls=tensor(3e5), afMean=1e-4,
             rrShape=tensor(50.), afShape=tensor(50.), pDs = None, generatingFn=v6normal,
             fitMethod='nelder-mead', nEpochs=20, mt=False,
             covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
             covSingle=tensor([[1, 0], [0, 1]]),
             nIterations=100,
             nEpochsPerIteration=1,
             runName="run",
             stacked=False,
             piPrior=False,
             old=False,
             rrtype="default"):
    print("rrtype", rrtype)
    import os
    from os import path
    from datetime import date
    import time
    import json

    params = []
    results = []

    folder = runName + "_" + str(date.today()) + str(int(time.time()))
    os.makedirs(folder, exist_ok=True)
    # allInferredParams = []
    # pis = []
    # alphas = []
    # pdv1true = []
    # pdv2true = []
    # pdv3true = []
    # pdv1inf = []
    # pdv2inf = []
    # pdv3inf = []
    print('covShared in runSimMT is', covShared)
    with Pool(cpu_count()) as p:
        y = 0
        for rrsSimRun in rrs:
            for pisSimRun in pis:
                paramsRun = genParams(rrMeans=rrsSimRun, pis=pisSimRun, afMean=afMean, pDs=pDs,
                                      rrShape=rrShape, afShape=afShape, nCases=nCases, nCtrls=nCtrls, covShared=covShared, covSingle=covSingle, rrtype=rrtype)[0]
                processors = []
                simRes = {"params": paramsRun, "runs": []}
                
                folder_inner = path.join(folder, f"{y}")
                # name = os.path.join(folder, "_".join([f"""{k}_{v}""" for (k, v) in paramsRun.items()]))
                os.makedirs(folder_inner, exist_ok=True)

                np.save(path.join(folder_inner, "params"), paramsRun)
                print("params are:", paramsRun)

                start = time.time()
                for i in range(nIterations):
                    # def runSimIteration(paramsRun, generatingFn=v6normal, fitMethod='Nelder-Mead', mt=False, nEpochs=1, stacked=False):
                    processors.append(p.apply_async(
                        processor, (i, paramsRun, {"generatingFn": generatingFn, "fitMethod": fitMethod, "mt": False, "nEpochs": nEpochsPerIteration, "stacked": stacked, "piPrior": piPrior, "old": old}), callback=lambda res: simRes["runs"].append(res)))
                # Wait for the asynchrounous reader threads to finish
                
                [r.get() for r in processors]

                print(f"finished sim of params: {paramsRun}")
                print(f"simulation took {time.time() - start}s")
                np.save(path.join(folder_inner, "data"), simRes)
                results.append(simRes)
                params.append(paramsRun)
                y += 1

                # br["pis"] = inferredPis
                # br["alphas"] = inferredAlphas
                # br["PDV_c1true"] = c1true
                # br["PDV_c2true"] = c2true
                # br["PDV_cBothTrue"] = cBothTrue
                # br["PDV_c1inferred"] = c1inferred
                # br["PDV_c2inferred"] = c2inferred
                # br["PDV_cBothInferred"] = cBothInferred
                # for res in simRes["runs"]:
                #     bestRes = res["results"]["bestRes"]
                #     pis.append(bestRes['pis'])
                #     alphas.append(bestRes['alphas'])
        
                #     pdv1true.append(bestRes['PDV_c1true'])
                #     pdv2true.append(bestRes['PDV_c2true'])
                #     pdv3true.append(bestRes['PDV_cBothTrue'])

                #     pdv1inf.append(bestRes['PDV_c1inferred'])
                #     pdv2inf.append(bestRes['PDV_c2inferred'])
                #     pdv3inf.append(bestRes['PDV_cBothInferred'])

                
                
        
        # print(results)
        # print("pdv3inf")

        # pis = tensor(pis)
        # alphas = tensor(alphas)

        # pdv1true = tensor(pdv1true)
        # pdv2true.append(bestRes['PDV_c2true'])
        # pdv3true.append(bestRes['PDV_cBothTrue'])

        # pdv1inf.append(bestRes['PDV_c1inferred'])
        # pdv2inf.append(bestRes['PDV_c2inferred'])
        # pdv3inf.append(bestRes['PDV_cBothInferred'])

        np.save(os.path.join(folder, "params_list"), params)
        np.save(os.path.join(folder, "results_list"), results)

        print("Done")

        return results

def runSimIteration(paramsRun, generatingFn=v6normal, fitMethod='Nelder-Mead', mt=False, nEpochs=1, stacked=False, piPrior=False, old=False):
    print("in runSimIteration params are", paramsRun)
    # In DSB:
    # 	No ID	ID
    #         ASD+ADHD	684	217
    #         ASD	3091	871
    #         ADHD	3206	271
    #         Control	5002	-

    #         gnomAD	44779	(Non-Finnish Europeans in non-psychiatric exome subset)

    #         Case total:	8340
    #         Control total:	49781
    # so we can use pDBoth = .1 * total_cases
    # needs tensor for shapes, otherwise "gamma_cpu not implemente for long", e.g rrShape=50.0 doesn't work...
    pDsRun = paramsRun["pDs"]
    pisRun = paramsRun["diseaseFractions"]
    afMeanRun = paramsRun["afMean"]
    rrMeansRun = paramsRun["rrMeans"]

    resToStore = {
        "allRes": None,
        "nEpochs": None,
        "bestRes": {
            "pis": None,
            "alphas": None,
            "PDV_c1true": None,
            "PDV_c2true": None,
            "PDV_cBothTrue": None,
            "PDV_c1inferred": None,
            "PDV_c2inferred": None,
            "PDV_cBothInferred": None,
        }
    }
   
    try:
        start = time.time()
        print(f"generating simulation using {generatingFn} using params: {paramsRun}")
        r = generatingFn(**paramsRun)
        print("took", time.time() - start)
    except Exception as e:
        print(f"Generating error: {e}")
        return {"error": e.__str__()}

    resPointer = {
        **r,
        "generatingFn": generatingFn,
        "results": None,
    }

    xsRun = resPointer["altCounts"]
    afsRun = resPointer["afs"]
    affectedGenesRun = resPointer["affectedGenes"]
    unaffectedGenesRun = resPointer["unaffectedGenes"]

    nSamples = tensor([ paramsRun["nCtrls"], *paramsRun["nCases"] ])
    runCostFnIdx = 0
    print("fit method is", fitMethod)
    print("nSamples are", nSamples)
    print("alt count sum are: genes1: ", xsRun[affectedGenesRun[0]].sum(0), "genes2: ", xsRun[affectedGenesRun[1]].sum(0), "genes3: ", xsRun[affectedGenesRun[2]].sum(0), "unaffectedGenes:", xsRun[unaffectedGenesRun].sum(0))
    print("alt count means are: genes1: ", xsRun[affectedGenesRun[0]].mean(0) / nSamples, "genes2: ", xsRun[affectedGenesRun[1]].mean(0) / nSamples, "genes3: ", xsRun[affectedGenesRun[2]].mean(0) / nSamples, "unaffectedGenes:", xsRun[unaffectedGenesRun].mean(0) / nSamples)
    start = time.time()
    if mt is True:
        res = fitFnBivariateMT(xsRun, pDsRun, nEpochs=nEpochs, minLLThresholdCount=20,
                               debug=True, costFnIdx=runCostFnIdx, method=fitMethod, stacked=stacked, piPrior=piPrior, old=old)
        bestRes = None
        bestLL = None
        for r in res:
            print("r:", r)
            if bestLL is None or r["lls"][-1] < bestLL:
                bestRes = r["params"][-1]
                bestLL = r["lls"][-1]
        print("bestLL", bestLL)
        print("bestRes", bestRes)
    else:
        # res here I think is different htan multi case
        if piPrior and stacked:
            res = fitFnBivariateStackedDirichlet(xsRun, pDsRun, nEpochs=nEpochs, minLLThresholdCount=20, debug=True)
        elif stacked:
            res = fitFnBivariateStacked(xsRun, pDsRun, nEpochs=nEpochs, minLLThresholdCount=20, debug=True)
        else:
            res = fitFnBivariate(xsRun, pDsRun, nEpochs=nEpochs, minLLThresholdCount=20,
                                debug=True, method=fitMethod,old=old)
        bestRes = res["params"][-1]
    print("took", time.time() - start)

    inferredPis = tensor(bestRes[0:3])  # 3-vector
    inferredAlphas = tensor(bestRes[3:])  # 4-vector, idx0 is P(!D|V)

    #### Calculate actual ###
    c1true, c2true, cBothTrue = empiricalPDGivenV(afsRun, affectedGenesRun, afMeanRun)

    # calculate inferred values
    c1inferred, c2inferred, cBothInferred = inferPDGivenVfromAlphas(inferredAlphas, pDsRun,old=old)

    print(f"\n\nrun results for rrs: {rrMeansRun}, pis: {pisRun}")

    print("Inferred pis:", inferredPis)
    print("\nP(D|V) true ans in component 1:", c1true)
    print("P(D|V) inferred in component 1:", c1inferred)
    print("\nP(D|V) true ans in component 1:", c2true)
    print("P(D|V) inferred in component both:", c2inferred)
    print("\nP(D|V) true ans in component both:", cBothTrue)
    print("P(D|V) inferred in component both:", cBothInferred, "\n\n")

    # this is too big, probably because of the trajectories
    del res["trajectoryLLs"]
    del res["trajectoryPi"]
    del res["trajectoryAlphas"]

    print("res", res)

    # TODO: write these to file somehow
    a2 = []
    for x in affectedGenesRun:
        a2.append([x[0], x[-1]])
    resPointer["affectedGenes"] = tensor(a2)
    resPointer["unaffectedGenes"] = tensor(
        [unaffectedGenesRun[0], unaffectedGenesRun[-1]])
    # TODO: figure out a better way to store  these
    del resPointer["rrs"]

    resToStore["allRes"] = res
    resToStore["nEpochs"] = nEpochs
    br = resToStore["bestRes"]
    br["pis"] = inferredPis
    br["alphas"] = inferredAlphas
    br["PDV_c1true"] = c1true
    br["PDV_c2true"] = c2true
    br["PDV_cBothTrue"] = cBothTrue
    br["PDV_c1inferred"] = c1inferred
    br["PDV_c2inferred"] = c2inferred
    br["PDV_cBothInferred"] = cBothInferred

    resPointer["results"] = resToStore

    return resPointer
