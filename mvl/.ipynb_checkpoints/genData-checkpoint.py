# import pyro
import torch
import torch.tensor as tensor
# import pyro.distributions as dist
from torch.distributions import Binomial, Gamma, Uniform
from pyro.distributions import Multinomial, Dirichlet

import numpy as np

import scipy
from skopt import gp_minimize 
from scipy.stats import binom as ScipyBinom
from matplotlib import pyplot

from collections import namedtuple

from .likelihoods import pVgivenD, pVgivenDapprox, pVgivenNotD, fitFnBivariate, fitFnBivariateMT
from pyper import *

import time
seed = 0

# Like the 4b case, but multinomial
# TODO: shoudl we do int() or some rounding function to go from float counts to int counts
def v5(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes = 20000):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape, "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)
    
    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape,rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape,rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000,]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))
    
    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes,])   
    rrs = rrDist.sample([nGenes,])

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
            rrSamples[2] = rrs[geneIdx, 0] #both always gets a rr of non-1
        elif affects == 2:
#             print(f"affects2: {geneIdx}")
            rrSamples[1] = rrs[geneIdx, 1]
            rrSamples[2] = rrs[geneIdx, 1]
        elif affects == 3:
#             print(f"affects2: {geneIdx}")
            rrSamples[0] = rrs[geneIdx, 0] + rrs[geneIdx, 2]
            rrSamples[1] = rrs[geneIdx, 1] + rrs[geneIdx, 2]
            rrSamples[2] = rrs[geneIdx, 0] + rrs[geneIdx, 1] + rrs[geneIdx, 2]
#         print("affects", affects, "rrSamples", rrSamples)
        probVgivenDs = pVgivenD(rrSamples, afs[geneIdx])
        probVgivenNotD = pVgivenNotD(pDs, afs[geneIdx], probVgivenDs)
        
        p=tensor([probVgivenNotD*(1-pDs.sum()), *(probVgivenDs*pDs)])
        
        totalProbability = p.sum()
#         print("af", afs[geneIdx], "probVgivenDs", probVgivenDs, "pDs", pDs, "probVgivenNotD", probVgivenNotD, "totalProbability", totalProbability)
        
        assert (abs(totalProbability-afs[geneIdx]) / afs[geneIdx]) < 1e-6
        marginalAlleleCount = int(totalProbability * totalSamples)
#         print("marginal allele count", marginalAlleleCount)
        
#         print("probs", probs)
        # without .numpy() can't later convert tensor(altCounts) : "only tensors can be converted to Python scalars"
        altCountsGene = Multinomial(probs=p, total_count=marginalAlleleCount).sample().numpy()
        
#         print("altCountsGene", altCountsGene)
        altCounts.append(altCountsGene)
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)
    
    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return { "altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll }

# Like 5, but make approximation that P(V|D) = P(V)*rr, by observing that rr*P(D|V) + 1-P(V) is ~1 for intermediate rr and small P(V)
# say a typical P(V|D) is ~
def v6(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes = 20000):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape, "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)
    
    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape,rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape,rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000,]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))
    
    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes,])   
    rrs = rrDist.sample([nGenes,])

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
#             print(f"affects2: {geneIdx}")
            rrSamples[0] = rrs[geneIdx, 0] + rrs[geneIdx, 2]
            rrSamples[1] = rrs[geneIdx, 1] + rrs[geneIdx, 2]
            rrSamples[2] = rrs[geneIdx, 0] + rrs[geneIdx, 1] + rrs[geneIdx, 2]
#         print("affects", affects, "rrSamples", rrSamples)
        probVgivenDs = pVgivenDapprox(rrSamples, afs[geneIdx])## rr*q
        probVgivenNotD = pVgivenNotD(pDs, afs[geneIdx], probVgivenDs)
        
        p=tensor([probVgivenNotD*(1-pDs.sum()), *(probVgivenDs*pDs)])
        
        totalProbability = p.sum()
#         print("af", afs[geneIdx], "probVgivenDs", probVgivenDs, "pDs", pDs, "probVgivenNotD", probVgivenNotD, "totalProbability", totalProbability)
#         print("abs(totalProbability-afs[geneIdx]) / afs[geneIdx]", abs(totalProbability-afs[geneIdx]) / afs[geneIdx])
        assert (abs(totalProbability-afs[geneIdx]) / afs[geneIdx]) <= 1e-6
        marginalAlleleCount = int(totalProbability * totalSamples)
#         print("marginal allele count", marginalAlleleCount)
        
#         print("probs", probs)
        # without .numpy() can't later convert tensor(altCounts) : "only tensors can be converted to Python scalars"
        altCountsGene = Multinomial(probs=p, total_count=marginalAlleleCount).sample().numpy()
        
#         print("altCountsGene", altCountsGene)
        altCounts.append(altCountsGene)
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)
    
    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return { "altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll }

# Like 6 but generates correlated relative risks by sampling from lognormal
def v6normal(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes = 20000,
             covShared=tensor([[1,.4,.4], [.4, 1, .4], [.4, .4, 1]]), covSingle=tensor([[1, 0], [0, 1]])):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape, "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)
    print("\n\ntest tensor",",".join(covShared.numpy().flatten()))
    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape,rate=afShape/afMean)
    
    r=R(use_pandas=True)
    r(f'''
        library(tmvtnorm)
        sigma <- matrix(c({",".join(torch.flatten(covShared))}), ncol={len(covShared)})
        rrsShared <- rtmvnorm(n={nGenes}, mean=c({rrMeans[0] + rrMeans[2]}, {rrMeans[1] + rrMeans[2]}, {rrMeans[0] + rrMeans[1] + rrMeans[2]}), sigma=sigma, lower=c(1,1,1))
        sigma <- matrix(c({",".join(torch.flatten(covSingle))}), ncol={len(covSingle)})
        rrsOne <- rtmvnorm(n={nGenes}, mean=c({rrMeans[0]}, {rrMeans[1]}), sigma=sigma, lower=c(1,1))
      ''')
    rrsShared = tensor(r.get('rrsShared'))
    rrsOne = tensor(r.get('rrsOne'))
    print(rrsShared)

    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes,])

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
        if affects == 0:
            unaffectedGenes.append(geneIdx)
        elif affects == 1:
            # TODO: do we need to have 0 correlation between rrSamples[0] and rrSampels[2]
            rrSamples[0] = rrsOne[geneIdx, 0]
            rrSamples[2] = rrSamples[0] 
        elif affects == 2:
            rrSamples[1] = rrsOne[geneIdx, 1]
            rrSamples[2] = rrSamples[1]
        elif affects == 3:
            rrSamples = rrsShared[geneIdx]
        
#         print("affects", affects, "rrSamples", rrSamples)

        probVgivenDs = pVgivenDapprox(rrSamples, afs[geneIdx])
        probVgivenNotD = pVgivenNotD(pDs, afs[geneIdx], probVgivenDs)
        
        p=tensor([probVgivenNotD*(1-pDs.sum()), *(probVgivenDs*pDs)])
        
        totalProbability = p.sum()
        
        assert (abs(totalProbability-afs[geneIdx]) / afs[geneIdx]) <= 1e-6
        marginalAlleleCount = int(totalProbability * totalSamples)

        altCountsGene = Multinomial(probs=p, total_count=marginalAlleleCount).sample().numpy()

        altCounts.append(altCountsGene)
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)
    
    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return { "altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll }

# Like 6, but only 2 groups of genes, those that affect 1only, or 2only. Samples that have both conditions just get rr1 in 1 genes, rr2 in 2 genes
# so the trick is that we have no 3rd component to infer; our algorithm should place minimal weight on that component
# if given 3 diseaseFractions, 3rd is ignored
def v6twoComponents(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes = 20000):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape, "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)
    
    diseaseFractions = diseaseFractions[0:-1] #bad, no reassign recommended
    nConditions = len(nCases) -1 
    assert(nConditions == 2)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape,rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape,rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000,]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))
    
    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes,])   
    rrs = rrDist.sample([nGenes,])

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
        rrSamples = tensor([1., 1., 1.]) #still 3, brecause we still have a "both" category
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
#         print("affects", affects, "rrSamples", rrSamples)
        probVgivenDs = pVgivenDapprox(rrSamples, afs[geneIdx])
        probVgivenNotD = pVgivenNotD(pDs, afs[geneIdx], probVgivenDs)
        
        p=tensor([probVgivenNotD*(1-pDs.sum()), *(probVgivenDs*pDs)])
        
        totalProbability = p.sum()
#         print("af", afs[geneIdx], "probVgivenDs", probVgivenDs, "pDs", pDs, "probVgivenNotD", probVgivenNotD, "totalProbability", totalProbability)
#         print("abs(totalProbability-afs[geneIdx]) / afs[geneIdx]", abs(totalProbability-afs[geneIdx]) / afs[geneIdx])
        assert (abs(totalProbability-afs[geneIdx]) / afs[geneIdx]) <= 1e-6
        marginalAlleleCount = int(totalProbability * totalSamples)
#         print("marginal allele count", marginalAlleleCount)
        
#         print("probs", probs)
        # without .numpy() can't later convert tensor(altCounts) : "only tensors can be converted to Python scalars"
        altCountsGene = Multinomial(probs=p, total_count=marginalAlleleCount).sample().numpy()
        
#         print("altCountsGene", altCountsGene)
        altCounts.append(altCountsGene)
        probs.append(p.numpy())
        rrAll.append(rrSamples)
    altCounts = tensor(altCounts)
    probs = tensor(probs)
    
    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return { "altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrAll }

# Like the 6 case, but we scale P(V|Ds) by prevalence, since the actual sample sizes say for the binomial in which P(V|D1) would be used is the fraction P(D1) of the total
# and in the multionmial setting, we use only a single sample size
# for instance, lets say we have 500k controls, 1000 cases
# the P(V|D) (cases) may be .0001 and P(V|!D) may  .0001, but the probability in a multinomial should really be 99.9999% in favor of controls
def v7(nCases, nCtrls, pDs, diseaseFractions, rrShape, rrMeans, afMean, afShape, nGenes = 20000):
    # TODO: assert shapes match
    print("TESTING WITH: nCases", nCases, "nCtrls", nCtrls, "rrMeans", rrMeans, "rrShape", rrShape, "afMean", afMean, "afShape", afShape, "diseaseFractions", diseaseFractions, "pDs", pDs)
    
    nConditions = len(nCases)
    assert(nConditions == 3)
    altCounts = []
    probs = []
    afDist = Gamma(concentration=afShape,rate=afShape/afMean)
    rrDist = Gamma(concentration=rrShape,rate=rrShape/rrMeans)
    print("rrDist mean", rrDist.sample([10_000,]).mean(0))
#     rrNullDist = Gamma(concentration=rrShape,rate=rrShape.expand(nConditions))
    
    # shape == [nGenes, nConditions]
    afs = afDist.sample([nGenes,])   
    rrs = rrDist.sample([nGenes,])

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

        PVNotD = pVgivenNotD(pDs, afs[geneIdx], PVDcases) #* (1 - pDs.sum())
        PVDcases = PVDcases #* pDs
            
        # P(D|V)/P(V)
        PVDprevalenceWeighted = PVDcases * pDs
        PVNotDprevalenceWeighted = PVNotD * (1 - pDs.sum())
        totalProbability = PVDprevalenceWeighted.sum() + PVNotDprevalenceWeighted
#         print("affects", affects, "af", afs[geneIdx], "PVDcases", PVDcases, "pDs", pDs, "PVNotD", PVNotD, "totalProbability", totalProbability)
        
        assert abs(totalProbability-afs[geneIdx]) / afs[geneIdx]  < 0.00001
        marginalAlleleCount = int(totalProbability * totalSamples)
#         print("marginal allele count", marginalAlleleCount)
        
        p=tensor([PVNotDprevalenceWeighted, *PVDprevalenceWeighted])
#         print("probs", probs)
        # without .numpy() can't later convert tensor(altCounts) : "only tensors can be converted to Python scalars"
        altCountsGene = Multinomial(probs=p, total_count=marginalAlleleCount).sample().numpy()
        
#         print("altCountsGene", altCountsGene)
        altCounts.append(altCountsGene)

        probs.append(p.numpy())
    altCounts = tensor(altCounts)
    probs = tensor(probs)
    
    # cannot convert affectedGenes to tensor; apparently tensors need to have same dimensions at each level of the tensor...stupid
    return { "altCounts": altCounts, "afs": probs, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "rrs": rrs }


def flattenAltCounts(altCounts, afs):
    altCountsFlatPooled = []
    afsFlatPooled = []
    for geneIdx in range(nGenes):
        altCountsFlatPooled.append([altCounts[geneIdx, 0, 0], *altCounts[geneIdx, :, 1].flatten()])
        afsFlatPooled.append([afs[geneIdx, 0, 0], *afs[geneIdx, :, 1].flatten()])

    altCountsFlatPooled = tensor(altCountsFlatPooled)
    afsFlatPooled = tensor(afsFlatPooled)
    print("altCountsFlatPooled", altCountsFlatPooled)
    print("afsFlatPooled", afsFlatPooled)

    flattenedData = []

    for geneAfData in afs:
        flattenedData.append([geneAfData[0][0],*geneAfData[:, 1]])
    flattenedData = tensor(flattenedData)
    
    return altCountsFlatPooled, afsFlatPooled, flattenedData

def genParams(pis = tensor([.1, .1, .05]), rrShape = tensor(10.), rrMeans = tensor([3., 3., 1.5]), afShape = tensor(10.), afMean = tensor(1e-4), nCases=tensor([5e3, 5e3, 2e3]), nCtrls=tensor(5e5)):
    nGenes = 20_000

    pDs = nCases / ( nCases.sum() + nCtrls )
    print("pDs are:", pDs)
    
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
    }]


import copy

def runSim(rrs = tensor([[1.5, 1.5, 1.5]]), pis = tensor([[.05, .05, .05]]), nCases = tensor([15e3, 15e3, 6e3]), nCtrls = tensor(5e5), afMean = 1e-4, rrShape=tensor(50.), afShape=tensor(50.), generatingFn =  v6normal, fitMethod = 'annealing', nEpochs=20, mt = False):
    resSim = {
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

    results = []
    print("MT is ", mt)
#     assert(1 == 0)
    i = 0
    for rrsSimRun in rrs:
        for pisSimRun in pis:
            
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
            paramsRun = genParams(rrMeans=rrsSimRun, pis=pisSimRun, afMean=afMean, rrShape=rrShape, afShape=afShape, nCases=nCases, nCtrls=nCtrls)[0]

            pDsRun = paramsRun["pDs"]
            pisRun = paramsRun["diseaseFractions"]
            print("params are:", paramsRun)

            results.append({"params": paramsRun, "runs": []})
            for y in range(0, 10):
                start = time.time()
                r = generatingFn(**paramsRun)
                print("took", time.time() - start)

                resPointer = {
                    **r,
                    "generatingFn": generatingFn,
                    "results": None,
                }

                results[i]["runs"].append(resPointer)

                print(f"Run: {i}, {y}")

                xsRun = resPointer["altCounts"]
                afsRun = resPointer["afs"]
                affectedGenesRun = resPointer["affectedGenes"]
                unaffectedGenesRun = resPointer["unaffectedGenes"]

                runCostFnIdx = 0
                print("fit method is", fitMethod)
                start = time.time()
                if mt is True:
                    res = fitFnBivariateMT(xsRun, pDsRun, nEpochs=nEpochs, minLLThresholdCount=20, debug=True, costFnIdx=runCostFnIdx, method=fitMethod)
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
                    res = fitFnBivariate(xsRun, pDsRun, nEpochs=nEpochs, minLLThresholdCount=20, debug=True, costFnIdx=runCostFnIdx, method=fitMethod)
                    bestRes = res["params"][-1]
                print("took", time.time() - start)

                inferredPis = tensor(bestRes[0:3]) # 3-vector
                inferredAlphas = tensor(bestRes[3:]) # 4-vector, idx0 is P(!D|V)

                #### Calculate actual ###
                component1Afs = afsRun[affectedGenesRun[0]]
                c1true = (component1Afs / afMean).mean(0)

                component2Afs = afsRun[affectedGenesRun[1]]
                c2true = (component2Afs / afMean).mean(0)

                componentBothAfs = afsRun[affectedGenesRun[2]]
                cBothTrue = (componentBothAfs / afMean).mean(0)

                ### calculate inferred values
                pds = tensor([1-pDsRun.sum(), *pDsRun])
                alphas = inferredAlphas.numpy()
                c1inferred = Dirichlet(tensor([alphas[0], alphas[1], alphas[0], alphas[2]]) * pds).sample([10_000]).mean(0)
                c2inferred = Dirichlet(tensor([alphas[0], alphas[0], alphas[2], alphas[2]]) * pds).sample([10_000]).mean(0)
                cBothInferred = Dirichlet(tensor([alphas[0], (alphas[1] + alphas[3]), (alphas[2] + alphas[3]), (alphas[1] + alphas[2] + alphas[3])]) * pds).sample([10_000]).mean(0)

                print(f"\n\nrun {i} results for rrs: {rrsSimRun}, pis: {pisSimRun}")
                print("Inferred pis:", inferredPis)
                print("\nP(D|V) true ans in component 1:", c1true)
                print("P(D|V) inferred in component 1:", c1inferred)
                print("\nP(D|V) true ans in component 1:", c2true)
                print("P(D|V) inferred in component both:", c2inferred)
                print("\nP(D|V) true ans in component both:", cBothTrue)
                print("P(D|V) inferred in component both:", cBothInferred,"\n\n")

                resToStore = copy.deepcopy(resSim)
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

            i += 1
        return results