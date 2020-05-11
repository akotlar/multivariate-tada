import pyro
import torch
import torch.tensor as tensor
import pyro.distributions as dist
# from torch.distributions import Binomial, Gamma, Uniform
from pyro.distributions import Binomial, Bernoulli, Categorical, Dirichlet, DirichletMultinomial, Beta, BetaBinomial, Uniform, Gamma, Multinomial

import numpy as np

import scipy
from skopt import gp_minimize 
from scipy.stats import binom as ScipyBinom
from matplotlib import pyplot

from collections import namedtuple
import time
seed = 0

#### Likelihood functions
# These assume univariate currently
 
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
        raise ValueError(f"pVgivenNotD: invalid params: pD: {pD}, pV: {pV}, pVgivenD: {pVgivenD} yield: p = {p}")
    return p

# def pVgivenNotD(pD, pV, pVgivenD):
#     p = (pV - (pD*pVgivenD)) / (1 - pD)
#     assert(p >= 0)
#     return p

def pDgivenV(pD, pVgivenD, pV):
    return pVgivenD * pD / pV

# works like shit
def llUnivariateSingleGeneJensen(xCtrl, xCase, pD, pi0, pi1, pDgivenV):
    n = xCtrl + xCase
    #convex function, so by jensen's sum of logs is fine (always <= the log of sum)
    return pi0 * Binomial(total_count=n, probs=pD).log_prob(xCase) + pi1*Binomial(total_count=n, probs=pDgivenV).log_prob(xCase)

def llUnivariateSingleGene(xCtrl, xCase, pD, pi0, pi1, pDgivenV):
    n = xCtrl + xCase
    #convex function, so by jensen's sum of logs is fine (always <= the log of sum)
    return torch.log(pi0 * torch.exp(Binomial(total_count=n, probs=pD).log_prob(xCase)) + pi1*torch.exp(Binomial(total_count=n, probs=pDgivenV).log_prob(xCase)))

# alphas shape: [2] #corresponding to cases and controls
def llUnivariateSingleGeneBetaBinomial(xCtrl, xCase, pD, alphas, pi0, pi1):
    n = xCtrl + xCase
    #convex function, so by jensen's sum of logs is fine (always <= the log of sum)
    # what is the 
    h0 = pi0 * torch.exp( Binomial(total_count=n, probs=pD).log_prob(xCase) )
    h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alphas[1], concentration0=alphas[0]).log_prob(xCase) )
    return torch.logalpha3( h0 + h1 )

# TODO: support pooled and non-pooled controls
# TODO: think about whether we need overlapping cases (both disease1 + disease2) or whether that can be inferred
# altCounts.shape = [1 control + nConditions cases, 1]
# alphas shape: [nConditions + 2] #1 ctrl + nCondition cases; for now the last condition in nCondition cases is for individuals who has all of the previous nConditions
# in a more multivariate setting we will need more information, aka mapping to which combinations of conditions these people have
# xCases: we have nConditions cases
# pDs shape: [nConditions]
# TODO: make this more effificent by taking alphas tensor of shape (1 + nConditions)
def llPooledBivariateSingleGene(altCounts, pDs, alpha0, alpha1, alpha2, alphaBoth, pi0, pi1, pi2, piBoth):
    # currently assume altCounts are all independent (in simulation), or 0 for everything but first condition
    n = altCounts.sum() 
    alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
    print("n is ", n)
    #convex function, so by jensen's sum of logs is fine (always <= the log of sum)
    # what is the 
    case1nullLikelihood = torch.exp( Binomial(total_count=n, probs=pDs[0]).log_prob(altCounts[1]) )
    case2nullLikelihood = torch.exp( Binomial(total_count=n, probs=pDs[1]).log_prob(altCounts[2]) )
    h0 = pi0 * case1nullLikelihood * case2nullLikelihood
    h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(altCounts[1]) ) * case2nullLikelihood
    h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(altCounts[2]) ) * case1nullLikelihood
    h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCounts))
    print(f"h0: {h0}, h1: {h1}, h2: {h2}, h3: {h3}")
    return torch.log( h0 + h1 + h2 + h3 )

# shape of altCountsByGene: [nGenes, nConditions, 2]
# last dimension is 
# 2nd dimension altCountsCasesByGene must match controls, or the control nConditions must be 1 (pooled controls)
def likelihoodUnivariate(altCountsByGene, pDs):
    nGenes = len(altCountsByGene)
    
    # passed to optimization function, we optimize pDgivenV and pi1 by maximizing likelihood
    def likelihood(params):
        pDgivenV = params[0]
        pi1 = params[1]
        pi0 = 1 - pi1
        
        if(pDgivenV >= 1 or pDgivenV < 0 or pi1 < 0 or pi1 > 1):
            print("returning inf")
            return float("-inf")
    
        logLikelihood = 0
        penaltyCount = float(nGenes)
        
        # 
        for geneIdx in range(nGenes):
            ctrlAltCount = altCountsByGene[geneIdx, 0, 0]
            caseAltCount = altCountsByGene[geneIdx, 0, 1]
            pd = pDs[0]
            
            if ctrlAltCount == 0 and caseAltCount == 0:
                print("skipping", geneIdx)
                continue

            # this is insanely slow
            ll = llUnivariateSingleGene(ctrlAltCount, caseAltCount, pd, pi0, pi1, pDgivenV)

            if torch.isnan(ll) or torch.isinf(ll):
                print(f"nan or 0 likelihood: like: {like}, p1: {pi1}, pDgivenV: {pDgivenV}, gene: {geneIdx}, ctrlCount: {ctrlAltCount}, caseCount: {caseAltCount}")
                penaltyCount -= 1
                continue
                
            logLikelihood += ll
        
    
        if penaltyCount == 0:
            penaltyCount = 1
    
        return -logLikelihood * (nGenes / penaltyCount)
    
    return likelihood

def likelihoodUnivariateFast(altCountsByGene, pDs):
    nGenes = len(altCountsByGene)
    geneSums = altCountsByGene[:, 0, :].sum(1)
        
    caseAltCounts = altCountsByGene[:, 0, 1]
    pD = pDs[0]
    def likelihood(params):
        pi1, pDgivenV = params

        pi0 = 1.0 - pi1

        if(pDgivenV > 1 or pDgivenV < 0 or pi1 < 0 or pi1 > 1):
            return float("inf")
        
        binomH0 = Binomial(total_count=geneSums, probs=pD)
        binomH1 = Binomial(total_count=geneSums, probs=pDgivenV)
        
        component0 = pi0 * torch.exp(binomH0.log_prob(caseAltCounts))
        component1 = pi1 * torch.exp(binomH1.log_prob(caseAltCounts))
        
        return - torch.log(component0 + component1).sum()
    
    return likelihood

def likelihoodUnivariateBetaBinomialFast(altCountsByGene, pDs):
    nGenes = len(altCountsByGene)
    geneSums = altCountsByGene[:, 0, :].sum(1)
        
    caseAltCounts = altCountsByGene[:, 0, 1]
    pD = pDs[0]
    def likelihood(params):
        pi1, alpha1, alpha0 = params

        if alpha1 < 0 or alpha0 < 0 or pi1 < 0 or pi1 > 1:
            return float("inf")
        
        pi0 = 1.0 - pi1

        binomH0 = Binomial(total_count=geneSums, probs=pD)
        binomH1 = BetaBinomial(total_count=geneSums, concentration1=alpha1, concentration0=alpha0)
        
        component0 = pi0 * torch.exp(binomH0.log_prob(caseAltCounts))
        component1 = pi1 * torch.exp(binomH1.log_prob(caseAltCounts))

        return - torch.log(component0 + component1).sum()
    
    return likelihood

def getUnivariateAlpha0(alpha1, pD):
    return ((1-pD) / pD)*alpha1

# doesn't really work constraint looks wrong
def likelihoodUnivariateBetaBinomialConstrainedFast(altCountsByGene, pDs):
    nGenes = len(altCountsByGene)
    geneSums = altCountsByGene[:, 0, :].sum(1)
        
    caseAltCounts = altCountsByGene[:, 0, 1]
    pD = pDs[0]
    pNotDRatio = (1 - pD)/pD
    def likelihood(params):
        pi1, alpha1 = params
        
        if alpha1 < 0 or pi1 < 0 or pi1 > 1:
            return float("inf")
        
        pi0 = 1.0 - pi1
        
        alpha0 = pNotDRatio*alpha1
        
        assert(alpha0 > 0)
        
        print("alpha0",alpha0)
        
        binomH0 = Binomial(total_count=geneSums, probs=pD)
        binomH1 = BetaBinomial(total_count=geneSums, concentration1=alpha1, concentration0=alpha0)
        
        component0 = pi0 * torch.exp(binomH0.log_prob(caseAltCounts))
        component1 = pi1 * torch.exp(binomH1.log_prob(caseAltCounts))

        return - torch.log(component0 + component1).sum()
    
    return likelihood

# Bivariate likelihood function modeled on:
#def llPooledBivariateSingleGene(altCounts, pDs, alpha0, alpha1, alpha2, alphaBoth, pi0, pi1, pi2, piBoth):
# # currently assume altCounts are all independent (in simulation), or 0 for everything but first condition
# n = altCounts.sum()
# alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
# case1nullLikelihood = torch.exp( Binomial(total_count=n, probs=pDs[0]).log_prob(altCounts[1]) )
# case2nullLikelihood = torch.exp( Binomial(total_count=n, probs=pDs[1]).log_prob(altCounts[2]) )
# h0 = pi0 * case1nullLikelihood * case2nullLikelihood
# h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(altCounts[1]) ) * case2nullLikelihood
# h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(altCounts[2]) ) * case1nullLikelihood
# h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCounts))
# print(f"h0: {h0}, h1: {h1}, h2: {h2}, h3: {h3}")
# return torch.log( h0 + h1 + h2 + h3 )
def likelihoodBivariateFast(altCountsByGene, pDs):
    print("shape", altCountsByGene.shape)
    nGenes = altCountsByGene.shape[0]
    
    if(altCountsByGene.shape[1] == 4):
        altCountsFlat = altCountsByGene
    else:
        altCountsFlat = []
        for geneIdx in range(nGenes):
            # ctrl count is first index of first condition, all other conditions get 0 count at 0th index
            altCountsFlat.append([altCountsByGene[geneIdx, 0, 0], *altCountsByGene[geneIdx, :, 1].flatten()])

    altCountsFlat = tensor(altCountsFlat)
    # nGenes x 4 
    xCtrl = altCountsFlat[:, 0]
    xCase1 = altCountsFlat[:, 1]
    xCase2 = altCountsFlat[:, 2]
    xCase12 = altCountsFlat[:, 3]
    # nGenes x 1
    n = xCtrl + xCase1 + xCase2 + xCase12
    print("altCountsFlat", altCountsFlat)
    print("n", n)
    print("xCase1, xCase2, xCase12", xCase1)
    print("xCase1, xCase2, xCase12", xCase2)
    print("xCase1, xCase2, xCase12", xCase12)
    
    pd1 = pDs[0]
    pd2 = pDs[1]
    pdBoth = pDs[2]
    
    # TODO: maybe we just want to explicitly use sample proportions
    pdCtrl = 1 - (pd1 + pd2 + pdBoth)

    case1Null = torch.exp(Binomial(total_count=n, probs=pd1).log_prob(xCase1))
    case2Null = torch.exp(Binomial(total_count=n, probs=pd2).log_prob(xCase2))
    caseBothNull = torch.exp(Binomial(total_count=n, probs=pdBoth).log_prob(xCase12))
    allNull = case1Null * case2Null * caseBothNull
    print("altCountsFlat", altCountsFlat)
    allNull2 = torch.exp(Multinomial(probs=tensor([1-pDs.sum(), pDs[0], pDs[1], pDs[2]])).log_prob(altCountsFlat))
    print("allNull2", allNull2)
    print("pd1, pd2, pdBoth, pdCtrl", pd1, pd2, pdBoth, pdCtrl)
    def likelihood1(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull

        h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(xCase1) ) * case2Null * caseBothNull
        h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(xCase2) ) * case1Null * caseBothNull
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat))

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood1a(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull

        h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(xCase1) ) * case2Null
        h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(xCase2) ) * case1Null
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat))

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood1b(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull

        h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(xCase1 + xCase12) ) * case2Null
        h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(xCase2 + xCase12) ) * case1Null
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat))

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihoodConstrained(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull
        
        # idea 1
        # alpha1 and alpha0 determined
        # A gene has counts from gene1 samples 2 , from gene2 samples 1 geneBoth count
        # if i have some people that only have 1, that is evidence for gene1 liability, but says nothing for liability for 
        # the more shared risk there is, the more the count will be in the "both category", 
        # the fewer people will be only one or the other
        # so eventually all 
        # 
        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(xCase1 + xCase12) ) * case2Null
        
        h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(xCase1 + xCase12) ) * case2Null
        h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(xCase2 + xCase12) ) * case1Null
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat))

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull

        h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1, concentration0=alphasSum - alpha1).log_prob(xCase1) ) * case2Null * torch.exp( BetaBinomial(total_count=n, concentration1=alphaBoth, concentration0=alphasSum - alphaBoth).log_prob(xCase12) )
        h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2, concentration0=alphasSum - alpha2).log_prob(xCase2) ) * case1Null * torch.exp( BetaBinomial(total_count=n, concentration1=alphaBoth, concentration0=alphasSum - alphaBoth).log_prob(xCase12) )
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat))

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2a(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull

        h1 = pi1 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha1 + alphaBoth, concentration0=alphasSum - alpha1).log_prob(xCase1) ) * case2Null
        h2 = pi2 * torch.exp( BetaBinomial(total_count=n, concentration1=alpha2 + alphaBoth, concentration0=alphasSum - alpha2).log_prob(xCase2) ) * case1Null 
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat))

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    
    def likelihood2b(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2

        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha0, alpha1])).log_prob(altCountsFlat) )
        h2 = pi2 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha0, alpha2, alpha2])).log_prob(altCountsFlat) )
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alphaBoth, alphaBoth, alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2c(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2

        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1+alphaBoth, alpha0, alpha1+alphaBoth])).log_prob(altCountsFlat) )
        h2 = pi2 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha0, alpha2+alphaBoth, alpha2+alphaBoth])).log_prob(altCountsFlat) )
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1+alphaBoth, alpha2+alphaBoth, alpha1+alphaBoth+alpha2])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2c(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2

        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha0, alpha1])).log_prob(altCountsFlat) )
        h2 = pi2 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha0, alpha2, alpha2])).log_prob(altCountsFlat) )
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2d(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBoth) / (1 - pDs[1])
        # h1 is that the genes in this component only increse risk for disease 1
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alpha1])).log_prob(altCountsFlat) )
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBoth) / (1 - pDs[0])
        # h2 is that the genes in this component only increse risk for disease 2
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alpha2])).log_prob(altCountsFlat) )
        # h3 is that the alleles in these genes increase risk for both diseases
        # we model by individual and shared components ; hower in this component do samples affected only by disease 1 necessarily have the probability of seeing
        # an allele that is the same as in the other cases? I think no, I think this is higher
        # as the shared component becomes large and larger, if it is added to the other alpha1 (the individual copmonet only), alpha1 will equal alphaBoth, i.e
        # they will be perfectly correlated
        # else, alpha0 will be some amount larger than alphaBoth
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2e(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        # P(D2)(others) / (1-PD2) = a2
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBoth) / (1 - pDs[1])
        # h1 is that the genes in this component only increse risk for disease 1
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alpha1])).log_prob(altCountsFlat) )
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBoth) / (1 - pDs[0])
        # h2 is that the genes in this component only increse risk for disease 2
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alpha2])).log_prob(altCountsFlat) )
        # h3 is that the alleles in these genes increase risk for both diseases
        # we model by individual and shared components ; hower in this component do samples affected only by disease 1 necessarily have the probability of seeing
        # an allele that is the same as in the other cases? I think no, I think this is higher
        # as the shared component becomes large and larger, if it is added to the other alpha1 (the individual copmonet only), alpha1 will equal alphaBoth, i.e
        # they will be perfectly correlated
        # else, alpha0 will be some amount larger than alphaBoth
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1 + alphaBoth, alpha2 + alphaBoth, alphaBoth + alpha1 + alpha2])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum
        
    # like 2d, but alphaBoth is the last component in each case
    # this allows ffor inference of a weighted alphaBoth
    # following the example of h3, where alpha1 is alpha1, not alpha1 + alphaBoth
    def likelihood2f(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBoth) / (1 - pDs[1])
        # h1 is that the genes in this component only increse risk for disease 1
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alphaBoth])).log_prob(altCountsFlat) )
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBoth) / (1 - pDs[0])
        # h2 is that the genes in this component only increse risk for disease 2
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alphaBoth])).log_prob(altCountsFlat) )
        # h3 is that the alleles in these genes increase risk for both diseases
        # we model by individual and shared components ; hower in this component do samples affected only by disease 1 necessarily have the probability of seeing
        # an allele that is the same as in the other cases? I think no, I think this is higher
        # as the shared component becomes large and larger, if it is added to the other alpha1 (the individual copmonet only), alpha1 will equal alphaBoth, i.e
        # they will be perfectly correlated
        # else, alpha0 will be some amount larger than alphaBoth
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
        
    # Like 2D, except alphaBoth in h1 and h2 is scaled by P(DBoth/PD1) or P(DBoth)/P(D1), since that is what is required
    # to go to P(D|V) given a shared P(V|D), which is all we mean when we try to equate alpha1 and alphaboth in component 1
    def likelihood2g(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBoth) / (1 - pDs[1])
        # h1 is that the genes in this component only increse risk for disease 1
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alpha1 * pDs[2] / pDs[0]])).log_prob(altCountsFlat) )
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBoth) / (1 - pDs[0])
        # h2 is that the genes in this component only increse risk for disease 2
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alpha2 * pDs[2] / pDs[1]])).log_prob(altCountsFlat) )
        # h3 is that the alleles in these genes increase risk for both diseases
        # we model by individual and shared components ; hower in this component do samples affected only by disease 1 necessarily have the probability of seeing
        # an allele that is the same as in the other cases? I think no, I think this is higher
        # as the shared component becomes large and larger, if it is added to the other alpha1 (the individual copmonet only), alpha1 will equal alphaBoth, i.e
        # they will be perfectly correlated
        # else, alpha0 will be some amount larger than alphaBoth
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alpha1 + alpha2 + alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
     # Like 2g, except follows the principle of the shared component being additive to individual components,
    # in the shared component
    def likelihood2h(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBoth) / (1 - pDs[1])
        alphaBothFrom1 = alpha1 * pDs[2] / pDs[0]
        # h1 is that the genes in this component only increse risk for disease 1
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alphaBothFrom1])).log_prob(altCountsFlat) )
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBoth) / (1 - pDs[0])
        alphaBothFrom2 = alpha2 * pDs[2] / pDs[1]
        # h2 is that the genes in this component only increse risk for disease 2
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alphaBothFrom2])).log_prob(altCountsFlat) )
        # h3 is that the alleles in these genes increase risk for both diseases
        # we model by individual and shared components ; hower in this component do samples affected only by disease 1 necessarily have the probability of seeing
        # an allele that is the same as in the other cases? I think no, I think this is higher
        # as the shared component becomes large and larger, if it is added to the other alpha1 (the individual copmonet only), alpha1 will equal alphaBoth, i.e
        # they will be perfectly correlated
        # else, alpha0 will be some amount larger than alphaBoth
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1 + alphaBoth, alpha2 + alphaBoth, alpha1 + alpha2 + alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
        
    # just let everything vary
    def likelihood2i(params):
        # TODO: better to do constrained or unconstrained alpha1?
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        # h1 is that the genes in this component only increse risk for disease 1
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) ) 
        # h2 is that the genes in this component only increse risk for disease 2
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) )
        # h3 is that the alleles in these genes increase risk for both diseases
        # we model by individual and shared components ; hower in this component do samples affected only by disease 1 necessarily have the probability of seeing
        # an allele that is the same as in the other cases? I think no, I think this is higher
        # as the shared component becomes large and larger, if it is added to the other alpha1 (the individual copmonet only), alpha1 will equal alphaBoth, i.e
        # they will be perfectly correlated
        # else, alpha0 will be some amount larger than alphaBoth
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
     # Like 2h except unfucked
    def likelihood2j(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        alphaBothFrom1 = alpha1 * pDs[2] / pDs[0]
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBothFrom1) / (1 - pDs[1])
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alphaBothFrom1])).log_prob(altCountsFlat) )
        
        alphaBothFrom2 = alpha2 * pDs[2] / pDs[1]
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBothFrom2) / (1 - pDs[0])
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alphaBothFrom2])).log_prob(altCountsFlat) )

        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1 + alphaBoth, alpha2 + alphaBoth, alpha1 + alpha2 + alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
     # Like 2j except allow last multinomial to freely vary as alpha1, alpha2, alphaBoth
    def likelihood2k(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
        
        alphasSum = alpha0 + alpha1 + alpha2 + alphaBoth
        
        h0 = pi0 * allNull2
        
        alphaBothFrom1 = alpha1 * pDs[2] / pDs[0]
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBothFrom1) / (1 - pDs[1])
        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alphaBothFrom1])).log_prob(altCountsFlat) )
        
        alphaBothFrom2 = alpha2 * pDs[2] / pDs[1]
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBothFrom2) / (1 - pDs[0])
        h2 = pi2 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alphaBothFrom2])).log_prob(altCountsFlat) )

        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2, alphaBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    def likelihood2l(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
                
        h0 = pi0 * allNull2
        
        alphaBothFrom1 = alpha1 * pDs[2] / pDs[0]
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBothFrom1) / (1 - pDs[1])
        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alphaBothFrom1])).log_prob(altCountsFlat) )
        
        alphaBothFrom2 = alpha2 * pDs[2] / pDs[1]
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBothFrom2) / (1 - pDs[0])
        h2 = pi2 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alphaBothFrom2])).log_prob(altCountsFlat) )
        
        alphaBoth1 = alphaBoth * pDs[0] / pDs[2]
        alphaBoth2 = alphaBoth * pDs[1] / pDs[2]
        alpha1_prime = alpha1 + alphaBoth1 #E[X_i] = n * alphaBoth / alphaSum
        alpha2_prime = alpha2 + alphaBoth2
        alphaBoth_prime = alphaBothFrom1 + alphaBothFrom2 + alphaBoth
        # Need cleaner walkthrough the different pieces; what is the prevalance piece vs scaling up or down that happens to match relative risk estimates
        # that it's a single effect size layered on the different proportions
        # we may want some multiplicative effect
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, alpha1_prime, alpha2_prime, alphaBoth_prime])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    # is_sparse with float32
    def likelihood2lSparse(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
                
        h0 = pi0 * allNull2
        
        alphaBothFrom1 = alpha1 * pDs[2] / pDs[0]
        # From E(P2_null) = P(D2) = a2 / (a0 + a1 + a2 + aB)
        alpha2Null = pDs[1] * (alpha0 + alpha1 + alphaBothFrom1) / (1 - pDs[1])
        h1 = pi1 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1, alpha2Null, alphaBothFrom1], dtype=torch.float32)).log_prob(altCountsFlat) )
        
        alphaBothFrom2 = alpha2 * pDs[2] / pDs[1]
        alpha1Null = pDs[0] * (alpha0 + alpha2 + alphaBothFrom2) / (1 - pDs[0])
        h2 = pi2 * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1Null, alpha2, alphaBothFrom2], dtype=torch.float32)).log_prob(altCountsFlat) )
        
        alphaBoth1 = alphaBoth * pDs[0] / pDs[2]
        alphaBoth2 = alphaBoth * pDs[1] / pDs[2]
        alpha1_prime = alpha1 + alphaBoth1 #E[X_i] = n * alphaBoth / alphaSum
        alpha2_prime = alpha2 + alphaBoth2
        alphaBoth_prime = alphaBothFrom1 + alphaBothFrom2 + alphaBoth
        # Need cleaner walkthrough the different pieces; what is the prevalance piece vs scaling up or down that happens to match relative risk estimates
        # that it's a single effect size layered on the different proportions
        # we may want some multiplicative effect
        h3 = piBoth * torch.exp( DirichletMultinomial(is_sparse=True, total_count=n, concentration=tensor([alpha0, alpha1_prime, alpha2_prime, alphaBoth_prime], dtype=torch.float32)).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()
    
    # In this model, by RWalter's suggestion, all of the alphas are latent, and so need to be scaled by the prevalence of the group
    def likelihood2m(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        
        if pi0 < 0:
            return float("inf")
     
        h0 = pi0 * allNull2

        actrl = pdCtrl * alpha0
        a11 = pd1 * alpha1
        a12 = pd2 * alpha0
        a1Both = pdBoth * alpha1
        
        h1 = pi1 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([actrl, a11, a12, a1Both])).log_prob(altCountsFlat) )
        
        a21 = pd1 * alpha0
        a22 = pd2 * alpha2
        a2Both = pdBoth * alpha2
        h2 = pi2 * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([actrl, a21, a22, a2Both])).log_prob(altCountsFlat) )
        
        aBoth1 = pd1 * (alpha1 + alphaBoth)
        aBoth2 = pd2 * (alpha2 + alphaBoth)
        aBothBoth = pdBoth * (alpha1 + alpha2 + alphaBoth)
        
        h3 = piBoth * torch.exp( DirichletMultinomial(total_count=n, concentration=tensor([alpha0, aBoth1, aBoth2, aBothBoth])).log_prob(altCountsFlat) )

        return -torch.log( h0 + h1 + h2 + h3 ).sum()

    return likelihood1, likelihood1a, likelihood1b, likelihood2, likelihood2a, likelihood2b, likelihood2c, likelihood2d, likelihood2e, likelihood2f, likelihood2g, likelihood2h, likelihood2i, likelihood2j, likelihood2k, likelihood2l, likelihood2m, likelihood2lSparse

def cb(f, context):
    print("got callback", f, context)

# TODO: update for multivariate
def fitFnUniveriate(altCountsByGene, pDs, nEpochs = 100, minLLThresholdCount = 100, debug = False):
    costFn = likelihoodUnivariateFast(altCountsByGene, pDs)
    
    lls = []
    params = []

    minLLDiff = 1
    thresholdHitCount = 0
    
    nGenes = len(altCountsByGene)

    randomDist = Uniform(1/nGenes, .5)
    randomDist2 = Uniform(0, 1)
    
        # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
    # P(V|D) * P(D) / P(V)
#     pDgivenVbounds = ( pVgivenD(2, 1e-6) * .001 / 1e-6, pVgivenD(100, 1e-2) * .1 / 1e-2 )
#     pi1Bounds = ( 1/nGenes,  1 )
#     bounds = [pDgivenVbounds, pi1Bounds]
    for i in range(nEpochs):
        best = float("inf")
        bestParams = []
        for y in range(100):
            # pi1, p(D|V)
            fnArgs = [randomDist.sample(), randomDist2.sample()]
            ll = costFn(fnArgs)
            if ll < best:
                best = ll
                bestParams = fnArgs
                
        if debug:
            print(f"best ll: {best}, params: {bestParams}")

        fit = scipy.optimize.minimize(costFn, x0 = bestParams, method='Nelder-Mead', options={"maxiter": 10000, "adaptive": True})#gp_minimize(costFn, [(1e-7, .9),(1/nGenes, .99)])#scipy.optimize.minimize(costFn, x0 = fnArgs, method="Nelder-Mead", options={"maxiter": 10000})
        
        if debug:
            print(f"epoch {i}")
            print(fit)

        if not fit["success"] is True:
            if debug:
                print("Failed to converge")
                print(fit)
            continue
        
        pi1, pDgivenV= fit["x"]
        if pDgivenV < 0 or pDgivenV > 1 or pi1 < 1/nGenes or pi1 > 1:
            if debug:
                print("Failed to converge")
                print(fit)
            continue
        
        ll = fit["fun"]
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
            
    return {"lls": lls, "params": params}


# TODO: maybe beta distribution should be constrained such that variance is that of the data?
# or maybe there's an analog to 0 mean liability variance?
def fitFnUniveriateBetaBinomial(altCountsByGene, pDs, nEpochs = 100, minLLThresholdCount = 100, debug = False):
    costFn = likelihoodUnivariateBetaBinomialFast(altCountsByGene, pDs)
    
    llsAll = []
    lls = []
    params = []

    minLLDiff = 1
    thresholdHitCount = 0
    
    nGenes = len(altCountsByGene)
    remainingEpochs = nEpochs
    
    randomDist = Uniform(1/nGenes, .5)
    randomDist2 = Uniform(100, 25000)
    # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
    # P(V|D) * P(D) / P(V)
    while remainingEpochs > 0:
        best = float("inf")
        bestParams = []
        for i in range(50):
            # pi1, alpha1, alpha0
            fnArgs = [randomDist.sample(), randomDist2.sample(), randomDist2.sample()]
            ll = costFn(fnArgs)
            if ll < best:
                best = ll
                bestParams = fnArgs
        
        if debug:
            print(f"best ll: {best}, bestParams: {bestParams}")

        fit = scipy.optimize.minimize(costFn, x0 = bestParams, method='Nelder-Mead', options={"maxiter": 10000, "adaptive": True})#gp_minimize(costFn, [(1e-7, .9),(1/nGenes, .99)])#scipy.optimize.minimize(costFn, x0 = fnArgs, method="Nelder-Mead", options={"maxiter": 10000})
        #fit = scipy.optimize.basinhopping(costFn, x0 = bestParams)
        if debug:
            print(f"epoch {remainingEpochs}")
            print(fit)

        if not fit["success"] is True:
            if debug:
                print("Failed to converge")
                print(fit)
            continue
        
        
        pi1, alpha1, alpha0 = fit["x"]
        # TODO: is pi1 > .5 restriction sound?
        if pi1 < 1/nGenes or pi1 > .5 or alpha1 <= 0 or alpha0 <= 0:
            if debug:
                print("Failed to converge")
                print(fit)
            continue
            
        remainingEpochs -= 1
        
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
            
    return {"lls": lls, "params": params, "llTrajectory": llsAll}

# TODO: maybe beta distribution should be constrained such that variance is that of the data?
# or maybe there's an analog to 0 mean liability variance
def fitFnBivariate(altCountsByGene, pDs, nEpochs = 100, minLLThresholdCount = 100, K = 4, debug = False, costFnIdx = 0):
    costFunctions = likelihoodBivariateFast(altCountsByGene, pDs)
        
    costFn = costFunctions[costFnIdx]
    print("past", costFn)
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
        # TODO: should we constrain alpha0 to the pD, i.e
        # E[P(D)] = alpha1 / sum(alphasRes)
        # P(D) * (alphasRes) = alpha1
        best = float("inf")
        bestParams = []
        for y in range(100):
            pi0 = pi0Dist.sample()
            pis = Uniform(1/nGenes, 1-pi0).sample([K-1])
            pis = pis/(pis.sum() + pi0)
#             print("pi0", pi0, "pis", pis, "sum", pis.sum())
            fnArgs = [*pis, *alphasDist.sample([K,])]

            ll = costFn(fnArgs)
            if ll < best:
                best = ll
                bestParams = fnArgs
        
        print(f"best ll: {best}, bestParams: {bestParams}")

#         fnArgs = [probs[0], probs[1], probs[2], *alphas]
        fit = scipy.optimize.minimize(costFn, x0 = bestParams, method='Nelder-Mead', options={"maxiter": 10000, "adaptive": True})

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
            
    return {"lls": lls, "params": params, "llTrajectory": llsAll}



def initBetaParams(mu, variance):
    alpha = ((1 - mu) / variance - 1 / variance) * mu**2
    beta = alpha * (1/mu -1)
    
    return alpha, beta