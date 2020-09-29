from torch.multiprocessing import cpu_count, Pool
# from pathos.multiprocessing import ProcessingPool as Pool
import dill

import torch
from torch import Tensor
import torch.tensor as tensor
from pyro.distributions import Binomial, Bernoulli, Categorical, Dirichlet, DirichletMultinomial, Beta, BetaBinomial, Uniform, Gamma, Multinomial, Gamma

import numpy as np

import scipy
from skopt import gp_minimize
from scipy.stats import binom as ScipyBinom
from matplotlib import pyplot

from collections import namedtuple
import time

from pyper import *
from torch import tensor
import torch
from . import optimize

r = R(use_pandas=True)

def skipColIdx(Y, idx):
    return torch.cat([Y[:,:~idx], Y[:,~idx:]], 1)


# m <- rowSums(Y)
    #     d <- ncol(Y)
        
    #     z <- t(apply(apply(apply(Y, 1, rev), 2, cumsum), 2, rev))
        
    #     logl <- (lgamma(m + 1) + rowSums(lgamma(Y[, -d] + alpha)) + 
    #       rowSums(lgamma(z[, -1] + beta)) + rowSums(lgamma(alpha + beta))) - 
    #       (rowSums(lgamma(Y + 1)) + rowSums(lgamma(alpha)) + rowSums(lgamma(beta)) + 
    #         rowSums(lgamma(alpha + beta + z[, -d])))

# Dirichlet Multinomial log prob, follows: https://rdrr.io/cran/MGLM/src/R/pdfln.R
def dgdirmn(Y, alpha, beta): 
    assert beta.shape == alpha.shape

    assert alpha.shape[0] == Y.shape[1] - 1
    alpha = alpha.expand((Y.shape[0], alpha.shape[0]))#     d <- ncol(Y)
    beta = beta.expand((Y.shape[0], beta.shape[0]))#     d <- ncol(Y)

    m = Y.sum(1) #m
    Yrev = Y.T.flip(0) 
    YrevCumsum = Yrev.cumsum(0) 
    z = YrevCumsum.flip(0).T

    n1 = torch.lgamma(m + 1)
    n2 = (torch.lgamma(Y[:, :-1] + alpha)).sum(1)
    n3 = (torch.lgamma(z[:, 1:] + beta)).sum(1)
    n4 = (torch.lgamma(alpha + beta)).sum(1)

    numerator = n1 + n2 + n3 + n4
    
    d1 = torch.lgamma(Y + 1).sum(1)
    d2 = torch.lgamma(alpha).sum(1)
    d3 = torch.lgamma(beta).sum(1)
    d4 = torch.lgamma(alpha + beta + z[:, :-1]).sum(1)

    denominator = d1 + d2 + d3 + d4

    return numerator - denominator

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

# def liabilityRisk(K):
#     K = integral(threshold to infiinity) { 1/sqrt(2pi) * exp(-x^2/2)dx}


# pD: prevalence, tensor of mConditions x 1
# pVgivenD: tensor of mConditions x 1
# pV: allele frequency

# rr = P(D|V) / P(D|~V)
# rr * P(D|~V) = P(D|V)
# P(V|D)P(D) = P(D|V)P(V)
# P(V|D) = rrP(D|~V)P(V) / P(D)
# let P(D) = rr * P(D|~V)P(V) + P(D|~V)P(~V)
# P(V|D) = rrP(V) / [ rrP(V) + ( 1 - P(V) ) ]

# in a null gene, P(D|~V) and P(D|V) are the same and P(D)
# probability of case status, and allele frequency are independent
# P(D) * P(V), P(D|V) * P(V)
# P(D|~V)P(~V) = P(~V|D)P(D)

def pVgivenD(rr, pV):
    return (rr * pV) / (rr * pV + (1 - pV))

def pVgivenDapprox(rr, pV):
    # P(V|Dboth) = (rr1 + rr2 + rrShared) * pV
    # (rr1*rr2) * pV
    return (rr * pV)

# I can't actually get this...
# i need everything in my study, oh well
# I'd need an estimate of P(V|D) for all the missing diseases
# def pVgivenNotDpop(pNotD, pV, pV)
#     pDall = 1 - pNotD
#     PDVall = pV

def pVgivenNotD(pD, pV, pVgivenD):
    p = (pV - (pD*pVgivenD).sum()) / (1 - pD.sum())
    if(p < 0):
        raise ValueError(
            f"pVgivenNotD: invalid params: pD: {pD}, pV: {pV}, pVgivenD: {pVgivenD}, (pD*pVgivenD).sum(): {(pD*pVgivenD).sum()} yield: p = {p}")
    return p
# P(D|V)P(V) == P(V|D)P(D)
def pNotDgivenVpV(PD: Tensor, PV: Tensor, PDV: Tensor):
    assert 1 - PD.sum() > 0
    p = PV - (PDV*PV).sum()
    if(p < 0):
        raise ValueError(
            f"pVgivenNotD: invalid params: pD: {PD}, pV: {PV}, PDV: {PDV}, (PV*PDV).sum(): {(PDV*PV).sum()} yield: p = {p}")
    return p

def getAltCountMeans(inData, params):
    samples = tensor([params["nCtrls"], *params["nCases"]])
    res = []
    
    res.append(inData["altCounts"][inData["unaffectedGenes"]].mean(0) / samples)

    for affectedGenes in inData["affectedGenes"]:
        res.append(inData["altCounts"][affectedGenes].mean(0) / samples)

    return res

# def dirichletPosterior(alphas, counts):

## Backign out to RR
# P(D1only|V,sample) = P(D1only,sample|V) / P(sample)          # by definition for conditional prob.
# = P(sample|D1only,V) P(D1only|V) / P(sample)                 # split out P of sampling vs D1
# = P(sample|D1only) P(D1only|V) / P(sample)                   # sample ascertainment doesn't depend on variants once conditioned on case status
# = [Ncase / pop_Ncase] P(D1only|V) / [N / pop_N]              # P(sample) defined in terms of # people in sample and in population
# = [Ncase / N] P(D1only|V) / [pop_Ncase / pop_N]              # rearrange
# = P(D1only|V) [sample_prevalence] / [population_prevalence]  # note quantities w/ N are prevalences
# ===> P(D1only|V) = P(D1only|V,sample) * [population_prevalence / sample_prevalence]
# P(D1|V) = P(D1only|V) + P(Dboth|V)
#  = P(D1only|V,sample) * [pop_prev_d1only / samp_prev_d1only] + P(Dboth|V,sample) * [pop_prev_dboth / samp_prev_dboth]

# RR = P(D|V)/P(D|~V)
# P(D) = P(D|V)P(V) + P(D|~V)P(~V)
# P(D) = P(D|V)P(V) + [P(D|V)/RR]P(~V)
# P(D) - P(D|V)P(V) = [P(D|V)/RR]P(~V)
# [P(D) - P(D|V)P(V)]/[P(D|V)P(~V)] = 1/RR
# RR = [P(D|V)P(~V)] / [P(D) - P(D|V)P(V)]

def pDgivenV(pD, pVgivenD, pV):
    return pVgivenD * pD / pV

def rrFromPD(pDgivenV, PD, PV):
    rr = pDgivenV * (1 - PV) / (PD - pDgivenV * PV)
    return rr

def dirichletMAP(alphas: Tensor, n: Tensor):
    denom = (alphas + n - 1).sum()
    return (alphas + n) / denom

def gdmMAP(alphas: Tensor, n: Tensor):
    denom = (alphas + n - 1).sum()
    return (alphas + n) / denom

# P(D_sample|V) * population_comorobidity / sample_comorbidity ~= (rr * P(V)) * P(D) / P(V) == rr * P(D)
# P(D|V) = [ (rr * pV) / (rr * pV + (1 - pV)) ] * P(D) / P(V)
# P(D|V) * P(V) / P(D) = rrPV / [rr * PV + (1-PV)]
# rr = P(D|V) / P(D|~V)

# Case1 = Binom(X1) + Binom(X3) ; Case2 = Binom(X2) + Binom(X3)

# rr1 , rr2 
def trueVsEst(inferred, input, params, old=False):
    nCases = params["nCases"]
    nCtrls = params["nCtrls"]
    samplePDs = nCases / (nCases.sum() + nCtrls)
    pDgivenVest = inferPDGivenVfromAlphas(
        tensor(inferred["params"][0][3:]), samplePDs=samplePDs, old=old)
    pDgivenVestVar = inferPDGivenVfromAlphasVar(
        tensor(inferred["params"][0][3:]), samplePDs=samplePDs, old=old)
    truePDGivenV = empiricalPDGivenV(
        input["afs"], affectedGenes=input["affectedGenes"], truePV=params["afMean"])

    print("est pis:", inferred["params"][0][0:3])
    print("tru pis:", params["diseaseFractions"])

    for i in range(len(pDgivenVest)):
        print(f"\n\nEstimate for component: {i}")
        print("est:", "P(D|V)", pDgivenVest[i], "variance:",
              pDgivenVestVar[i], "alphas:", pDgivenVest[i])
        print("tru:", "P(D|V)", truePDGivenV[i], "alphas:", truePDGivenV[i])

    return pDgivenVest, pDgivenVestVar, truePDGivenV

def trueVsEstA0(inferred, input, params):
    nCases = params["nCases"]
    nCtrls = params["nCtrls"]
    samplePDs = nCases / (nCases.sum() + nCtrls)
    pDgivenVest = inferPDGivenVfromAlphasA0(
        tensor(inferred["params"][0][3:]), samplePDs=samplePDs)
    pDgivenVestVar = inferPDGivenVfromAlphasVarA0(
        tensor(inferred["params"][0][3:]), samplePDs=samplePDs)
    truePDGivenV = empiricalPDGivenV(
        input["afs"], affectedGenes=input["affectedGenes"], truePV=params["afMean"])

    print("est pis:", inferred["params"][0][0:3])
    print("tru pis:", params["diseaseFractions"])

    for i in range(len(pDgivenVest)):
        print(f"\n\nEstimate for component: {i}")
        print("est:", "P(D|V)", pDgivenVest[i], "variance:",
              pDgivenVestVar[i], "alphas:", pDgivenVest[i])
        print("tru:", "P(D|V)", truePDGivenV[i], "alphas:", truePDGivenV[i])

    return pDgivenVest, pDgivenVestVar, truePDGivenV

def getDirichlets(alphasTensor, samplePDs):
    pdsAll = tensor([1-samplePDs.sum(), *samplePDs])
    alphas = alphasTensor.numpy()
    c1inferred = Dirichlet(tensor(
        [alphas[0], alphas[1], alphas[0], alphas[1]]) * pdsAll)
    c2inferred = Dirichlet(tensor(
        [alphas[0], alphas[0], alphas[2], alphas[2]]) * pdsAll)
    cBothInferred = Dirichlet(tensor([alphas[0], alphas[1], alphas[2], alphas[3]]) * pdsAll)

    return c1inferred, c2inferred, cBothInferred

def getDirichletsOld(alphasTensor, samplePDs):
    print("callind OLD")
    pdsAll = tensor([1-samplePDs.sum(), *samplePDs])
    alphas = alphasTensor.numpy()
    c1inferred = Dirichlet(tensor(
        [alphas[0], alphas[1], alphas[0], alphas[1]]) * pdsAll)
    c2inferred = Dirichlet(tensor(
        [alphas[0], alphas[0], alphas[2], alphas[2]]) * pdsAll)
    cBothInferred = Dirichlet(tensor([alphas[0], (alphas[1] + alphas[3]), (alphas[2] + alphas[3]),
                                      (alphas[1] + alphas[2] + alphas[3])]) * pdsAll)

    return c1inferred, c2inferred, cBothInferred

def getDirichletsA0(alphasTensor, samplePDs):
    pdsAll = tensor([1-samplePDs.sum(), *samplePDs])
    alphas = alphasTensor.numpy()
    c1inferred = Dirichlet(tensor(
        [alphas[0], alphas[0] + alphas[1], alphas[0], alphas[0] + alphas[1]]) * pdsAll)
    c2inferred = Dirichlet(tensor(
        [alphas[0], alphas[0], alphas[0] + alphas[2], alphas[0] + alphas[2]]) * pdsAll)
    cBothInferred = Dirichlet(tensor([alphas[0], (alphas[0] + alphas[1] + alphas[3]), (alphas[0] + alphas[2] + alphas[3]),
                                      (alphas[0] + alphas[1] + alphas[2] + alphas[3])]) * pdsAll)

    return c1inferred, c2inferred, cBothInferred

def inferPDGivenVfromAlphas(alphasTensor, samplePDs, old=False):
    if old:
        c1inferred, c2inferred, cBothInferred = getDirichletsOld(alphasTensor, samplePDs)
    else:
        c1inferred, c2inferred, cBothInferred = getDirichlets(alphasTensor, samplePDs)
    print(c1inferred.mean.numpy())
    return [c1inferred.mean.numpy(), c2inferred.mean.numpy(), cBothInferred.mean.numpy()]

def inferPDGivenVfromAlphasA0(alphasTensor, samplePDs):
    c1inferred, c2inferred, cBothInferred = getDirichletsA0(alphasTensor, samplePDs)
    print(c1inferred.mean.numpy())
    return [c1inferred.mean.numpy(), c2inferred.mean.numpy(), cBothInferred.mean.numpy()]


def inferPDGivenVfromAlphasVar(alphasTensor, samplePDs, old=False):
    if old:
        c1inferred, c2inferred, cBothInferred = getDirichletsOld(alphasTensor, samplePDs)
    else:
        c1inferred, c2inferred, cBothInferred = getDirichlets(alphasTensor, samplePDs)
    return [c1inferred.variance.numpy(), c2inferred.variance.numpy(), cBothInferred.variance.numpy()]

def inferPDGivenVfromAlphasVarA0(alphasTensor, samplePDs):
    c1inferred, c2inferred, cBothInferred = getDirichletsA0(alphasTensor, samplePDs)
    return [c1inferred.variance.numpy(), c2inferred.variance.numpy(), cBothInferred.variance.numpy()]

# TODO: generalize to N components
# truePV = true allele frequency


def empiricalPDGivenV(afs, affectedGenes, truePV):
    component1Afs = afs[affectedGenes[0]]
    c1true = (component1Afs / truePV).mean(0)

    component2Afs = afs[affectedGenes[1]]
    c2true = (component2Afs / truePV).mean(0)

    componentBothAfs = afs[affectedGenes[2]]
    cBothTrue = (componentBothAfs / truePV).mean(0)
    return [c1true.numpy(), c2true.numpy(), cBothTrue.numpy()]


def getAlphas(fit):
    return tensor(fit["params"][0][3:])


def getPis(fit):
    return tensor(fit["params"][0][:3])


def nullLikelihoodLog(pDsAll, altCounts):
    print("pDS are", pDsAll)
    print("altCounts are", altCounts)
    return Multinomial(probs=pDsAll).log_prob(altCounts)


def nullLikelihood(pDsAll, altCounts):
    return torch.exp(nullLikelihoodLog(pDsAll, altCounts))

# TODO: separate individual gene counts
# and joint gene counts
# 
def effectLLDMD(nHypotheses, pDs, altCountsFlat):
    nGenes = altCountsFlat.shape[0]
    nConditions = altCountsFlat.shape[1]
    nHypothesesNonNull = nHypotheses - 1

    # nGenes x 1
    n = altCountsFlat.sum(1) #xCtrl + xCase1 + xCase2 + xCase12
    print("n", n)
    altCountsShaped = altCountsFlat.expand(
        nHypothesesNonNull, nGenes, nConditions).transpose(0, 1)

    altCountsShaped = altCountsFlat.expand(nHypothesesNonNull, nGenes, nConditions).transpose(0, 1)
    altCountsShapedRepeat = torch.repeat_interleave(altCountsShaped, torch.tensor([1, 1, 2]), 1)  
    nShaped = n.expand(nHypothesesNonNull, nGenes).T
    nShapedRepeat = n.expand(nHypothesesNonNull + 1, nGenes).T

    pdsAll = tensor([1 - pDs.sum(), *pDs])
    pdsAllShaped = pdsAll.expand(nHypothesesNonNull, nConditions)
    pdsAllShapedRepeat = pdsAll.expand(nHypothesesNonNull + 1, nConditions)

    def dmd(alphas, betas):
        # total sums to 1
        # and P(D0|V)P(V) + P(D1|V)P(V)
        concentrations = pdsAllShaped * tensor([
            #ctrls, cases1, cases2, casesBoth
            # keep reuse; to reflec the fact that we're not truly multinomial
            # these are really independent binomials

            [alpha0, alpha1, alpha0, alpha1],  # H1 alpha1/(alpha0 + alpha1 + alpha2 + alpha1)
            [alpha0, alpha0, alpha2, alpha2],  # H2
            # alpha_sum = a0 + a1 + aBoth + a2 + aBoth + a1 + a2 + aBoth
            # E(P(V|D)P(V)) = alpha1 + alphaBoth / alpha_sum
            [alpha0, alpha1, alpha2, alphaBoth]  # H1&2&3 P(V|D1) alpha1/(alpha0 + alpha1 + alpha2 + alphaBoth)
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

    return {
        "dmd": dmd,
    }

# TODO: separate individual gene counts
# and joint gene counts
# 
def effectLikelihood(nHypotheses, pDs, altCountsFlat, nCases: Tensor, nCtrls: Tensor):
    nGenes = altCountsFlat.shape[0]
    nConditions = altCountsFlat.shape[1]
    nHypothesesNonNull = nHypotheses - 1

    # nGenes x 1
    n = altCountsFlat.sum(1) #xCtrl + xCase1 + xCase2 + xCase12
    print("n", n)
    altCountsShaped = altCountsFlat.expand(
        nHypothesesNonNull, nGenes, nConditions).transpose(0, 1)

    altCountsShaped = altCountsFlat.expand(nHypothesesNonNull, nGenes, nConditions).transpose(0, 1)
    altCountsShapedRepeat = torch.repeat_interleave(altCountsShaped, torch.tensor([1, 1, 2]), 1)  
    nShaped = n.expand(nHypothesesNonNull, nGenes).T
    nShapedRepeat = n.expand(nHypothesesNonNull + 1, nGenes).T

    samplePDs = nCases / (nCases.sum() + nCtrls)
    pdsAll = tensor([1 - samplePDs.sum(), *samplePDs])
    print("pdsAll", pdsAll)
    pdsAllPop = tensor([1 - pDs.sum(), *pDs])
    print("pdsall", pdsAllPop)
    # pdsAll = pdsAllPop
    pdsAllShaped = pdsAll.expand(nHypothesesNonNull, nConditions)
    pdsAllShapedRepeat = pdsAll.expand(nHypothesesNonNull + 1, nConditions)

    # if we were to have more models, we would have something like k-(k-1)! more models, or k-1more models
    # [alpha0, alpha0, alpha0, alpha3, alpha3], #H3
    # covariance
    # alpha11 =
    # need to use family structure to estimate covariance matrix beyond 2 conditions
    # [alpha0, alpha0, alpha0, alpha3 + alpha1, alpha3 + alpha1], #H3
    # H1&3
    # H2&3
    # H1&2
    # H1&2&3
    def likelihoodFn2(alpha0, alpha1, alpha2):
        concentrations = pdsAllShaped[0:2, :] * tensor([
            [alpha0, alpha1, alpha0, alpha1],  # H1
            [alpha0, alpha0, alpha2, alpha2],  # H2
        ]).expand(nGenes, 2, 2)

        # try to stay in the log space; avoid numeric underflow, until bayes factor calc
        return torch.exp(DirichletMultinomial(total_count=nShaped[:, 2], concentration=concentrations).log_prob(altCountsShaped[:, 0:3]))

    # alphas give me probability of Disease and Variant (exposure)
    # I could alternatively of supply probability of disease from population prevalence
    # for disease alone and disease both, that would let me get back to a relative  risk that is a relative risk 
    # P(D_population)/P(D_in_sample)
    # Once I have that, I can rebalance case both, case single disease
    # 
    # CONNECTION to fisher exact test
    # Issue: how does this parameterizations of the alphas translate to relative risks
    def likelihoodFn(alpha0, alpha1, alpha2, alphaBoth):
        # total sums to 1
        # and P(D0|V)P(V) + P(D1|V)P(V)
        concentrations = pdsAllShaped * tensor([
            #ctrls, cases1, cases2, casesBoth
            # keep reuse; to reflec the fact that we're not truly multinomial
            # these are really independent binomials

            [alpha0, alpha1, alpha0, alpha1],  # H1 alpha1/(alpha0 + alpha1 + alpha2 + alpha1)
            [alpha0, alpha0, alpha2, alpha2],  # H2
            # alpha_sum = a0 + a1 + aBoth + a2 + aBoth + a1 + a2 + aBoth
            # E(P(V|D)P(V)) = alpha1 + alphaBoth / alpha_sum
            [alpha0, alpha1, alpha2, alphaBoth]  # H1&2&3 P(V|D1) alpha1/(alpha0 + alpha1 + alpha2 + alphaBoth)
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

        # where correlation may live:
        # P(D1|V) = P(D1|V,architecture1only)*P(architecture1only) + P(D1|V,comorbid_architecture)*P(comorbid_architecture)
        # * this is handled only in-sample; need something to get back to population; in some semblance of populatoin 
        # relative: IPsych ; we have a lot more than probaiblity disease both
        # * he wants thje relative likelihood of being a case
        # Observed in-sample P(D1only|V) = P(V|D1)*rr1only*P(V) and P(Dboth|V) = P(V|DBoth)*rrBoth*P(V)
        # P(D1|v) = P(D1only|V) + P(DBoth|V)
        # If I take the in-sample P(D1|V), that is the 


        #### P(D1|V)
        # population_comorbidity_rate = #cases_comorbidity / #cases_no_comorbidity
        # P(D1|V) = P(D1|V,architecture1)*P(architecture1)*P(D_population1only)/in_sample_case1proprotion * p + P(D1|V,architectureBoth)*population_comorbidity_rate / (case1/case1+casesBoth)
        # is architecture proportion related to case comorbidity rate? it seems it must be, because if 
        # the only genetic architecture was the one that contributed to both disease,
        # comorbidity rate would be 100% (without any environmental variance) if we assumed 
        # equal contribution to both conditions, and if not,  it would be 
        # the mean contribution * proportion of genetic architecture
        # mean contribution here is alpha1 for disease1
        # ... if alphaBoth indexes risk both then the comorbidity rate would be 
        # something proportional to alphaBoth * piBoth right?
        # what is the ratio alpha1*pi1/alphaBoth*piBoth
        # ok, then the ratio alpha1/alphaBoth indexes the relative contribution to either 1 or the comorbid case, on average
        # and pi1/piBoth indexes the likelihood of that genetic architecture... but that's meaningless in isolation I think
        # fully depends on the effect ratio.
        # 
        # Ah yes, so there is a balance between risk increasing to 1only, 2only, or both
        # If a gene is truly risk increasing to both
        # we should get no contribution to 1only or 2only
        # and how large alphaBoth is relative to alpha1 gives us a measure of that; it almost seems like piBoth and pi1 not needed
        # so I'd say that comorbidity rate is something like aBoth*piBoth/a1*pi1
        # and there will be a natural tension between the relative proportion of cases both and cases 1 and the likelihood that we call a gene having architecture 1 only
        # 
        # What is the relation between the relative risks?
        # Well, rr1 only and rrBoth I think both increase risk of disease 1
        # so P(D1|V) = P(D1only|V)P(V) + P(D1and2|V)
        # # in the way that P(A) or P(B) = P(A) + P(B) - P(AandB)
        # so P(A) + P(B) = P(A) or P(B) + P(AandB)
        # and P(A) = P(A or B) + P(A and B) - P(B)
        # and P(A) = P(A) + P(AandB)
        # and P(B) = P(B) + P(AandB)
        # call D1 = A
        # call D2 = B
        # P(D1) = P(D1) + P(D1andD2)
        # call D1andD2 DBoth
        # P(D1) = P(D1) + P(DBoth)
        # P(D1) = P(D1|V)P(V) + P(D1|~V)P(~V) + P(DBoth|V)P(V) + P(DBoth|~V)P(~V)
        # call P(V) the probability that the contribution comes from a gene that is risk-increasing
        # call P(~V) the probability of getting a variant from a null gene (variant does nothing)
        # P(V) + P(~V) = 1 
        # 1 = P(V|D)P(D) + P(V|~D)P(~D) + P(~V|D)P(D) + P(~V|~D)P(~D)
        # P(~V) = 1-P(V) = 1 - P(V|D)P(D) - P(V|~D)P(~D)
        # We have P(V) an P(~V)

        # 

        # print("concentrations", concentrations)
        # print("altCountsShaped", altCountsShaped)
        # Binom(p1) + Binom(pshared) = Binom(p1 + pshared)
        # Covariance
        # Cov(Y1, Y2) ; where Y1 = X1 + XShared, Y2 = X2 + XShared X1 ~ Binom(p1), X2 ~ Binom(p2), XShared ~ Binom(pshared)
        #  = Variance(XShared) = n * pShared(1-pShared)

        # Results:
        # est of alpha1; I take the expectation of this for P(D|V)P(V) /  = P(D|V); I have P(D)P(V), 
        # try to stay in the log space; avoid numeric underflow, until bayes factor calc
        return torch.exp(DirichletMultinomial(total_count=nShaped, concentration=concentrations).log_prob(altCountsShaped))

        # Issue: how does this parameterizations of the alphas translate to relative risks
    def likelihoodFnOld(alpha0, alpha1, alpha2, alphaBoth):
        # total sums to 1
        # and P(D0|V)P(V) + P(D1|V)P(V)
        concentrations = pdsAllShaped * tensor([
            #ctrls, cases1, cases2, casesBoth
            [alpha0, alpha1, alpha0, alpha1],  # H1
            [alpha0, alpha0, alpha2, alpha2],  # H2
            # alpha_sum = a0 + a1 + aBoth + a2 + aBoth + a1 + a2 + aBoth
            # E(P(V|D)P(V)) = alpha1 + alphaBoth / alpha_sum
            [alpha0, alpha1 + alphaBoth, alpha2 + alphaBoth, alpha1 + alpha2 + alphaBoth]  # H1&2&3
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

        # Binom(p1) + Binom(pshared) = Binom(p1 + pshared)
        # Covariance
        # Cov(Y1, Y2) ; where Y1 = X1 + XShared, Y2 = X2 + XShared X1 ~ Binom(p1), X2 ~ Binom(p2), XShared ~ Binom(pshared)
        #  = Variance(XShared) = n * pShared(1-pShared)

        # try to stay in the log space; avoid numeric underflow, until bayes factor calc
        return torch.exp(DirichletMultinomial(total_count=nShaped, concentration=concentrations).log_prob(altCountsShaped))

    def likelihoodFnAlphaEverywhere(alpha0, alpha1, alpha2, alphaBoth):
        concentrations = pdsAllShaped * tensor([
            [alpha0, alpha0 + alpha1, alpha0, alpha0 + alpha1],  # H1
            [alpha0, alpha0, alpha0 + alpha2, alpha0 + alpha2],  # H2
            [alpha0, alpha0 + alpha1 + alphaBoth, alpha0 + alpha2 + alphaBoth,
                alpha0 + alpha1 + alpha2 + alphaBoth]  # H1&2&3
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

        # try to stay in the log space; avoid numeric underflow, until bayes factor calc
        return torch.exp(DirichletMultinomial(total_count=nShaped, concentration=concentrations).log_prob(altCountsShaped))

    def likelihoodFnSimpleNoLatent(alpha0, alpha1, alpha2, *args):
        concentrations = pdsAllShaped * tensor([
            [alpha0, alpha1, alpha0, alpha1],  # H1
            [alpha0, alpha0, alpha2, alpha2],  # H2
            [alpha0, alpha1, alpha2, alpha1 + alpha2]  # H1&2&3
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

        # try to stay in the log space; avoid numeric underflow, until bayes factor calc
        return torch.exp(DirichletMultinomial(total_count=nShaped, concentration=concentrations).log_prob(altCountsShaped))
    
    def likelihoodFnBoth(alpha0, alpha1, alpha2, alphaBoth):
        concentrations = pdsAllShapedRepeat * tensor([
            [alpha0, alpha1, alpha0, alpha1],  # H1
            [alpha0, alpha0, alpha2, alpha2],  # H2
            [alpha0, alpha1 + alphaBoth, alpha2 + alphaBoth, alpha1 + alpha2 + alphaBoth],  # H1&2&3
            [alpha0, alpha1, alpha2, alpha1 + alpha2]  # H1&2&3
        ]).expand(nGenes, nHypothesesNonNull + 1, nConditions)

        # try to stay in the log space; avoid numeric underflow, until bayes factor calc
        return torch.exp(DirichletMultinomial(total_count=nShapedRepeat, concentration=concentrations).log_prob(altCountsShapedRepeat))

    return likelihoodFn, nullLikelihood(pdsAll, altCountsFlat), likelihoodFnSimpleNoLatent, likelihoodFnBoth, likelihoodFn2, likelihoodFnAlphaEverywhere, likelihoodFnOld

def effectLikelihood3(nHypotheses, pDs, altCountsFlat):
    assert(nHypotheses == 7) #ctrl, case1, case2, case3, case12, case23, case123
    nHypothesesNonNull = nHypotheses - 1

    print("IN: altCountsFlat", altCountsFlat.shape)
    nGenes = altCountsFlat.shape[0]
    nConditions = altCountsFlat.shape[1]

    # pd1, pd2, pd3, pd12, pd23, pd123 = pDs

    pdCtrl = 1 - pDs.sum()

    print("pDs: ", pDs)

    # nGenes x 4
    # xCtrl = altCountsFlat[:, 0:] #, xCase1, xCase2, xCase3, xCase12, xCase23, xCase123 
    # xCase1 = altCountsFlat[:, 1]
    # xCase2 = altCountsFlat[:, 2]
    # xCase3 = altCountsFlat[:, 3]
    # xCase12 = altCountsFlat[:, 4]
    # xCase23 = altCountsFlat[:, 5]
    # xCase123 = altCountsFlat[:, 6]
    # nGenes x 1
    n = altCountsFlat.sum(1)
    print(n)

    altCountsShaped = altCountsFlat.expand(
        nHypothesesNonNull, nGenes, nConditions).transpose(0, 1)
    nShaped = n.expand(nHypothesesNonNull, nGenes).T
    pdsAllShaped = pDs[1:].expand(nHypothesesNonNull, nConditions)

    def formLikelihoods(a0, a1, a2, a3, a12, a13, a23, a123):
        t = tensor([
            #ctrl, case1, case2, case3, case12, case13, case23, case123
            [a0, a1, a0, a0, a1, a1, a0, a1],  # H1 only
            [a0, a0, a2, a0, a2, a0, a2, a2],  # H2
            [a0, a0, a0, a3, a0, a3, a3, a3],  # H3
            [a0, a1, a2, a0, a12, a1, a2, a12],  # affects both 1 and 2
            [a0, a1 + a13, a0, a3 + a13, a1 + a13, a1 + a3 + a13, a3 + a13, a1 + a13],  # affects both 1 and 3
            [a0, a0, a2 + a23, a3 + a23, a2 + a23, a3 + a23, a2 + a3 + a23, a2 + a3 + a23],  # affects both 2 and 3
            [a0, a1 + a123, a2 + a123, a3 + a123, a1 + a2 + a123, a1 + a3 + a123, a2 + a3 + a123, a1 + a2 + a3 + a123],  # affects 1 and 2 and 3
        ]).expand(nGenes, nHypothesesNonNull, nConditions)

    # if we were to have more models, we would have something like k-(k-1)! more models, or k-1more models
    # [alpha0, alpha0, alpha0, alpha3, alpha3], #H3
    # covariance
    # alpha11 =
    # need to use family structure to estimate covariance matrix beyond 2 conditions
    # [alpha0, alpha0, alpha0, alpha3 + alpha1, alpha3 + alpha1], #H3
    # H1&3
    # H2&3
    # H1&2
    # H1&2&3
    # def likelihoodFn(a0, a1, a2, a3, a12, a13, a23, a123):
    #     concentrations = pdsAllShaped * 

    #     # try to stay in the log space; avoid numeric underflow, until bayes factor calc
    #     return torch.exp(DirichletMultinomial(total_count=nShaped, concentration=concentrations).log_prob(altCountsShaped))

    # return likelihoodFn, nullLikelihood(pDs, altCountsFlat)


def likelihoodBivariateFast(altCountsFlat, pDs, nCases: Tensor, nCtrls: Tensor, trajectoryPis, trajectoryAlphas, trajectoryLLs):
    print(altCountsFlat.shape)

    # TODO: make this flexible for multivariate
    nHypotheses = 4
    nGenes = altCountsFlat.shape[0]
  
    likelihoodFn, allNull2, likelihoodFnNoLatent, likelihoodBothModels, likelihoodFn2, likelihoodFnA0, likelihoodFnOld = effectLikelihood(
        nHypotheses, pDs, altCountsFlat, nCases, nCtrls)

    def jointLikelihood2(params):
        pi1, pi2, alpha0, alpha1, alpha2 = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or pi1 < 0 or pi2 < 0:
            return float("inf")

        pi0 = 1.0 - (pi1 + pi2)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2])
        trajectoryAlphas.append([alpha0, alpha1, alpha2])

        hs = tensor([[pi1, pi2]]) * \
            likelihoodFn2(alpha0, alpha1, alpha2)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll

    # I estimate 1 set of genome-wide alphas
    # but once I have this, I can go back to the per-gene observations
    # and say given this is the maximized model (pis, alphas), waht is our
    # # expectation for the 
    # def jointLikelihood(params):
    #     pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

    #     if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
    #         print("returning inf")
    #         return float("inf")

    #     pi0 = 1.0 - (pi1 + pi2 + piBoth)
    #     # print("pi0")
    #     if pi0 < 0:
    #         return float("inf")

    #     h0 = pi0 * allNull2

    #     trajectoryPis.append([pi1, pi2, piBoth])
    #     trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

    #     hs = tensor([[pi1, pi2, piBoth]]) * likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)

    #     ll = -torch.log(h0 + hs.sum(1)).sum()
    #     # print("ll", ll)
    #     trajectoryLLs.append(ll)
    #     return ll
    
    # like above but constrains pseudo-counts
    def jointLikelihood(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")
        
        # TODO: Figure out how to reliably prefer smaller pseudocounts
        # if alpha1 > 1e6 or alpha2 > 1e6 or alphaBoth > 1e6:
        #     print("returning inf due to alphas", params)
        #     return float("inf")

        pi0 = 1.0 - (pi1 + pi2 + piBoth)
        # print("pi0")
        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2, piBoth])
        trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

        hs = tensor([[pi1, pi2, piBoth]]) * likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        # print("ll", ll)
        trajectoryLLs.append(ll)
        return ll

    # like the above, but without exploding "params"
    # def jointLikelihood(params):
    #     # pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

    #     for param in params:
    #         if param < 0:
    #             return float("inf")

    #     # if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
    #     #     print("returning inf")
    #     #     return float("inf")

    #     pi0 = 1.0 - sum(params[0:3])
    #     # print("pi0")
    #     if pi0 < 0:
    #         return float("inf")

    #     h0 = pi0 * allNull2

    #     trajectoryPis.append(params[0:3])
    #     trajectoryAlphas.append(params[3:])

    #     hs = tensor(params[0:3]) * likelihoodFn(*params[3:])

    #     ll = -torch.log(h0 + hs.sum(1)).sum()
    #     # print("ll", ll)
    #     trajectoryLLs.append(ll)
    #     return ll

    def jointLikelihoodOld(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")

        pi0 = 1.0 - (pi1 + pi2 + piBoth)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2, piBoth])
        trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

        hs = tensor([[pi1, pi2, piBoth]]) * likelihoodFnOld(alpha0, alpha1, alpha2, alphaBoth)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll

    def jointLikelihoodA0(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")

        pi0 = 1.0 - (pi1 + pi2 + piBoth)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2, piBoth])
        trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

        hs = tensor([[pi1, pi2, piBoth]]) * \
            likelihoodFnA0(alpha0, alpha1, alpha2, alphaBoth)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll

    def jointLikelihoodSimple(params):
        pi1, pi2, piBoth, alpha0, alpha1, alpha2 = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
            return float("inf")

        pi0 = 1.0 - (pi1 + pi2 + piBoth)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2, piBoth])
        trajectoryAlphas.append([alpha0, alpha1, alpha2])

        hs = tensor([[pi1, pi2, piBoth]]) * \
            likelihoodFnNoLatent(alpha0, alpha1, alpha2)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll

    def jointLikelihoodBoth(params):
        pi1, pi2, piBoth, piBothSimple, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0 or piBothSimple < 0:
            return float("inf")

        pi0 = 1.0 - (pi1 + pi2 + piBoth + piBothSimple)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append([pi1, pi2, piBoth, piBothSimple])
        trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

        hs = tensor([[pi1, pi2, piBoth, piBothSimple]]) * \
            likelihoodBothModels(alpha0, alpha1, alpha2, alphaBoth)

        ll = -torch.log(h0 + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll

    def jointLikelihoodDirichlet(params):
        pi0, pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

        if alpha0 < 1 or alpha1 < 1 or alpha2 < 1 or alphaBoth < 1 or pi0 < 1 or pi1 < 1 or pi2 < 1 or piBoth < 1:
            return float("inf")

        raise Exception("BLAH")

        pis = Dirichlet(tensor([pi0, pi1, pi2, piBoth])).mean
        trajectoryPis.append(pis)
        trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])
        # print("in joint likelihood")
        null = pis[0] * allNull2
        hs = likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)
        # print("hs", hs)
        ll = -torch.log(null + hs.sum(1)).sum()
        trajectoryLLs.append(ll)
        return ll

    def jointLikelihoodSimpleDirichlet(params):
        pi0, pi1, pi2, piBoth, alpha0, alpha1, alpha2 = params

        if alpha0 < 1 or alpha1 < 1 or alpha2 < 1 or pi0 < 1 or pi1 < 1 or pi2 < 1 or piBoth < 1:
            return float("inf")
        
        raise Exception("BLAH")
        pis = Dirichlet(tensor([pi0, pi1, pi2, piBoth])).mean
        # print('pis', pis)
        # print(" pis[:, 1:].shape",  pis[:, 1:].shape)

        trajectoryPis.append(pis)
        trajectoryAlphas.append([alpha0, alpha1, alpha2])
        r = likelihoodFnNoLatent(alpha0, alpha1, alpha2)
        # print("likelihood", r, "shape", r.shape)
        null = pis[0] * allNull2
        # print("null", null)
        hs = (pis[1:] * r).sum(1)
        # print("hs", hs)
        ll = -torch.log(null + hs).sum()
        trajectoryLLs.append(ll)
        return ll

    return jointLikelihood, jointLikelihoodSimple, jointLikelihoodBoth, jointLikelihood2, jointLikelihoodDirichlet, jointLikelihoodSimpleDirichlet, jointLikelihoodA0, jointLikelihoodOld

def likelihoodDMD(altCountsFlat, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs):
    nGenes = altCountsFlat.shape[0]
    funcs = effectLLDMD(pDs, altCountsFlat)

    dmd = funcs["dmd"]

    def ll(params):
        for param in params:
            if param < 0:
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

    return {
        "ll": ll
    }


# def likelihoodBivariateFast(altCountsFlat, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs):
#     print(altCountsFlat.shape)

#     # TODO: make this flexible for multivariate
#     nHypotheses = 4
#     nGenes = altCountsFlat.shape[0]
#     print("nGenes")
#     likelihoodFn, allNull2, likelihoodFnNoLatent, likelihoodBothModels, likelihoodFn2, likelihoodFnA0, likelihoodFnOld = effectLikelihood(
#         nHypotheses, pDs, altCountsFlat)

#     def jointLikelihood2(params):
#         pi1, pi2, alpha0, alpha1, alpha2 = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or pi1 < 0 or pi2 < 0:
#             return float("inf")

#         pi0 = 1.0 - (pi1 + pi2)

#         if pi0 < 0:
#             return float("inf")

#         h0 = pi0 * allNull2

#         trajectoryPis.append([pi1, pi2])
#         trajectoryAlphas.append([alpha0, alpha1, alpha2])

#         hs = tensor([[pi1, pi2]]) * \
#             likelihoodFn2(alpha0, alpha1, alpha2)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     # I estimate 1 set of genome-wide alphas
#     # but once I have this, I can go back to the per-gene observations
#     # and say given this is the maximized model (pis, alphas), waht is our
#     # expectation for the 
#     def jointLikelihood(params):
#         pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
#             return float("inf")

#         pi0 = 1.0 - (pi1 + pi2 + piBoth)

#         if pi0 < 0:
#             return float("inf")

#         h0 = pi0 * allNull2

#         trajectoryPis.append([pi1, pi2, piBoth])
#         trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

#         hs = tensor([[pi1, pi2, piBoth]]) * likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     def jointLikelihoodOld(params):
#         pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
#             return float("inf")

#         pi0 = 1.0 - (pi1 + pi2 + piBoth)

#         if pi0 < 0:
#             return float("inf")

#         h0 = pi0 * allNull2

#         trajectoryPis.append([pi1, pi2, piBoth])
#         trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

#         hs = tensor([[pi1, pi2, piBoth]]) * likelihoodFnOld(alpha0, alpha1, alpha2, alphaBoth)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     def jointLikelihoodA0(params):
#         pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
#             return float("inf")

#         pi0 = 1.0 - (pi1 + pi2 + piBoth)

#         if pi0 < 0:
#             return float("inf")

#         h0 = pi0 * allNull2

#         trajectoryPis.append([pi1, pi2, piBoth])
#         trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

#         hs = tensor([[pi1, pi2, piBoth]]) * \
#             likelihoodFnA0(alpha0, alpha1, alpha2, alphaBoth)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     def jointLikelihoodSimple(params):
#         pi1, pi2, piBoth, alpha0, alpha1, alpha2 = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0:
#             return float("inf")

#         pi0 = 1.0 - (pi1 + pi2 + piBoth)

#         if pi0 < 0:
#             return float("inf")

#         h0 = pi0 * allNull2

#         trajectoryPis.append([pi1, pi2, piBoth])
#         trajectoryAlphas.append([alpha0, alpha1, alpha2])

#         hs = tensor([[pi1, pi2, piBoth]]) * \
#             likelihoodFnNoLatent(alpha0, alpha1, alpha2)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     def jointLikelihoodBoth(params):
#         pi1, pi2, piBoth, piBothSimple, alpha0, alpha1, alpha2, alphaBoth = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi2 < 0 or piBoth < 0 or piBothSimple < 0:
#             return float("inf")

#         pi0 = 1.0 - (pi1 + pi2 + piBoth + piBothSimple)

#         if pi0 < 0:
#             return float("inf")

#         h0 = pi0 * allNull2

#         trajectoryPis.append([pi1, pi2, piBoth, piBothSimple])
#         trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

#         hs = tensor([[pi1, pi2, piBoth, piBothSimple]]) * \
#             likelihoodBothModels(alpha0, alpha1, alpha2, alphaBoth)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     def jointLikelihoodDirichlet(params):
#         pi0, pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = params

#         if alpha0 < 1 or alpha1 < 1 or alpha2 < 1 or alphaBoth < 1 or pi0 < 1 or pi1 < 1 or pi2 < 1 or piBoth < 1:
#             return float("inf")

#         raise Exception("BLAH")

#         pis = Dirichlet(tensor([pi0, pi1, pi2, piBoth])).mean
#         trajectoryPis.append(pis)
#         trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])
#         # print("in joint likelihood")
#         null = pis[0] * allNull2
#         hs = likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)
#         # print("hs", hs)
#         ll = -torch.log(null + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     def jointLikelihoodSimpleDirichlet(params):
#         pi0, pi1, pi2, piBoth, alpha0, alpha1, alpha2 = params

#         if alpha0 < 1 or alpha1 < 1 or alpha2 < 1 or pi0 < 1 or pi1 < 1 or pi2 < 1 or piBoth < 1:
#             return float("inf")
        
#         raise Exception("BLAH")
#         pis = Dirichlet(tensor([pi0, pi1, pi2, piBoth])).mean
#         # print('pis', pis)
#         # print(" pis[:, 1:].shape",  pis[:, 1:].shape)

#         trajectoryPis.append(pis)
#         trajectoryAlphas.append([alpha0, alpha1, alpha2])
#         r = likelihoodFnNoLatent(alpha0, alpha1, alpha2)
#         # print("likelihood", r, "shape", r.shape)
#         null = pis[0] * allNull2
#         # print("null", null)
#         hs = (pis[1:] * r).sum(1)
#         # print("hs", hs)
#         ll = -torch.log(null + hs).sum()
#         trajectoryLLs.append(ll)
#         return ll

#     return jointLikelihood, jointLikelihoodSimple, jointLikelihoodBoth, jointLikelihood2, jointLikelihoodDirichlet, jointLikelihoodSimpleDirichlet, jointLikelihoodA0, jointLikelihoodOld

def likelihood3(altCountsFlat, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs):
    nHypotheses = 7
    nGenes = altCountsFlat.shape[0]

    likelihoodFn, allNull2 = effectLikelihood3(nHypotheses, pDs, altCountsFlat)

    def jointLikelihood(params):
        pis = params[0:6]
        alphas = params[6:]

        if any(x < 0 for x in params):
            return float("inf")

        pi0 = 1.0 - sum(pis)

        if pi0 < 0:
            return float("inf")

        h0 = pi0 * allNull2

        trajectoryPis.append(pis)
        trajectoryAlphas.append(alphas)

        hs = tensor([pis]) * likelihoodFn(*alphas)

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


def processorStacked(i, *args, **kwargs):
    np.random.seed()
    torch.manual_seed(np.random.randint(1e9))
    r = fitFnBivariateStacked(*args, **kwargs)
    return r


def processorStackedDirichlet(i, *args, **kwargs):
    np.random.seed()
    torch.manual_seed(np.random.randint(1e9))
    r = fitFnBivariateStackedDirichlet(*args, **kwargs)
    return r


def fitFnBivariateMT(altCountsFlat, pDs, nCases: Tensor, nCtrls: Tensor, nEpochs=20, minLLThresholdCount=100, K=4, debug=False, stacked=False, piPrior=False):
    args = [altCountsFlat, pDs, nCases, nCtrls, 1, minLLThresholdCount,
            K, debug]

    results = []

    with Pool(cpu_count()) as p:
        processors = []
        for i in range(nEpochs):
            if piPrior and stacked:
                processors.append(p.apply_async(processorStackedDirichlet, (i, *args), callback=lambda res: results.append(res)))
            elif stacked:
                processors.append(p.apply_async(processorStacked, (i, *args), callback=lambda res: results.append(res)))
            else:
                processors.append(p.apply_async(processor, (i, *args), callback=lambda res: results.append(res)))
        # Wait for the asynchrounous reader threads to finish
        [r.get() for r in processors]

        return results

# TODO: maybe beta distribution should be constrained such that variance is that of the data?
# or maybe there's an analog to 0 mean liability variance


def minimizerr(costFn, x0, kwargs):
    # fun, args = dill.loads(costFn)
    # return fun(*args)
    print(kwargs)
    return scipy.optimize.minimize(costFn, x0, **kwargs)

# def run_dill_encoded(payload):
#     fun, args = dill.loads(payload)
#     return fun(*args)


# def apply_async(pool, fun, args):
#     payload = dill.dumps((fun, args))
#     return pool.apply_async(run_dill_encoded, (payload,))

def fitFnBivariateGDM(altCountsByGene, pDs, nEpochs=1, minLLThresholdCount=100, K=4, debug=False, method="nelder-mead", old=False):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    trajectoryPisSimple = []
    trajectoryAlphasSimple = []
    trajectoryLLsSimple = []
    costFn, costFnSimple, costFnBoth, _, _, _, costFnA0, costFnOld = likelihoodBivariateFast(
        altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)

    assert(method == "nelder-mead" or method ==
           "annealing" or method == "basinhopping")

    llsAll = []
    lls = []
    params = []

    minLLDiff = 1
    thresholdHitCount = 0

    nGenes = len(altCountsByGene)
    print("In main function")
    # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
    # P(V|D) * P(D) / P(V)
    pi0Dist = Uniform(.5, 1)
    alphasDist = Uniform(100, 25000)
    fitSimple = None

    if old:
        costFn = costFnOld

    print("method", method, "costFn", costFn)

    for i in range(nEpochs):
        start = time.time()

        if method == "nelder-mead" or method == "basinhopping":
            best = float("inf")
            bestParams = []
            for y in range(10):
                pi0 = pi0Dist.sample()
                pis = Uniform(1/nGenes, 1-pi0).sample([K-1])
                pis = pis/(pis.sum() + pi0)
                fnArgs = [*pis.numpy(), *alphasDist.sample([K, ]).numpy()]

                ll = costFn(fnArgs)
                if ll < best:
                    best = ll
                    bestParams = fnArgs

            if method == "nelder-mead":
                print("Running single-step optimization")
                fit = minimizerr(costFn, bestParams, {
                                    "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})
            elif method == "basinhopping":
                fit = scipy.optimize.basinhopping(costFn, x0=bestParams)
            else:
                raise Exception("should have been nelder-mead or basinhopping")
        elif method == "annealing":
            fit = scipy.optimize.dual_annealing(costFn, [(
                0.001, .999), (.001, .999), (.001, .999), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000)])

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

def fitFnBivariate(altCountsByGene, pDs, nCases: Tensor, nCtrls: Tensor, nEpochs=1, minLLThresholdCount=100, K=4, debug=False, method="nelder-mead", old=False):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    trajectoryPisSimple = []
    trajectoryAlphasSimple = []
    trajectoryLLsSimple = []
    costFn, costFnSimple, costFnBoth, _, _, _, costFnA0, costFnOld = likelihoodBivariateFast(
        altCountsByGene, pDs, nCases, nCtrls, trajectoryPis, trajectoryAlphas, trajectoryLLs)
    print('costFn', costFn)
    assert(method == "nelder-mead" or method ==
           "annealing" or method == "basinhopping")

    llsAll = []
    lls = []
    params = []

    minLLDiff = 1
    thresholdHitCount = 0

    nGenes = len(altCountsByGene)
    print("In main function")
    # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
    # P(V|D) * P(D) / P(V)
    pi0Dist = Uniform(.5, 1)
    alphasDist = Uniform(1, 250)
    fitSimple = None

    if old:
        costFn = costFnOld

    print("method", method, "costFn", costFn)

    for i in range(nEpochs):
        start = time.time()

        if method == "nelder-mead" or method == "basinhopping":
            best = float("inf")
            bestParams = []
            for y in range(10):
                pi0 = pi0Dist.sample()
                pis = Uniform(1/nGenes, 1-pi0).sample([K-1])
                pis = pis/(pis.sum() + pi0)
                fnArgs = [*pis.numpy(), *alphasDist.sample([K, ]).numpy()]
                print("fnArgs", fnArgs)
                ll = costFn(fnArgs)
                print("ll", ll)
                if ll < best:
                    best = ll
                    bestParams = fnArgs
            print("bestParams", bestParams)
            if method == "nelder-mead":
                print("Running single-step optimization")
                fit = minimizerr(costFn, bestParams, {
                                    "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})
            elif method == "basinhopping":
                fit = scipy.optimize.basinhopping(costFn, x0=bestParams)
            else:
                raise Exception("should have been nelder-mead or basinhopping")
        elif method == "annealing":
            fit = scipy.optimize.dual_annealing(costFn, [(
                0.001, .999), (.001, .999), (.001, .999), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000)])

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

def fitFnBivariateA0(altCountsByGene, pDs, nEpochs=1, minLLThresholdCount=100, K=4, debug=False, method="nelder-mead"):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    trajectoryPisSimple = []
    trajectoryAlphasSimple = []
    trajectoryLLsSimple = []
    costFn, costFnSimple, costFnBoth, _, _, _, costFnA0 = likelihoodBivariateFast(
        altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)
    print("method", method, "costFn", costFn)

    assert(method == "nelder-mead" or method ==
           "annealing" or method == "basinhopping")

    llsAll = []
    lls = []
    params = []

    minLLDiff = 1
    thresholdHitCount = 0

    nGenes = len(altCountsByGene)
    print("In main function")
    # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
    # P(V|D) * P(D) / P(V)
    pi0Dist = Uniform(.5, 1)
    alphasDist = Uniform(100, 25000)
    fitSimple = None
    for i in range(nEpochs):
        start = time.time()

        if method == "nelder-mead" or method == "basinhopping":
            best = float("inf")
            bestParams = []
            for y in range(10):
                pi0 = pi0Dist.sample()
                pis = Uniform(1/nGenes, 1-pi0).sample([K-1])
                pis = pis/(pis.sum() + pi0)
                fnArgs = [*pis.numpy(), *alphasDist.sample([K, ]).numpy()]

                ll = costFnA0(fnArgs)
                if ll < best:
                    best = ll
                    bestParams = fnArgs

            if method == "nelder-mead":
                print("Running single-step optimization")
                fit = minimizerr(costFnA0, bestParams, {
                                    "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})
            elif method == "basinhopping":
                fit = scipy.optimize.basinhopping(costFn, x0=bestParams)
            else:
                raise Exception("should have been nelder-mead or basinhopping")
        elif method == "annealing":
            fit = scipy.optimize.dual_annealing(costFn, [(
                0.001, .999), (.001, .999), (.001, .999), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000)])

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

# like fitFnBivariate, but slowly builds up the full model,
# inferring parameters for the lower-dimensional models first
# then using those as starting point for the bigger model
def fitFnBivariateStacked(altCountsByGene, pDs, K=4, debug=False, **kwargs):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    trajectoryPisSimple = []
    trajectoryAlphasSimple = []
    trajectoryLLsSimple = []
    costFn, costFnSimple, costFnBoth, costFn2, _, _= likelihoodBivariateFast(
        altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)

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
    fitSimple = None


    best = float("inf")
    bestParamsSimple = []
    for y in range(10):
        pi0 = pi0Dist.sample()
        pis = Uniform(1/nGenes, 1-pi0).sample([K-1])
        pis = pis/(pis.sum() + pi0)
        fnArgs = [*pis.numpy(), *alphasDist.sample([K-1, ]).numpy()]

        ll = costFnSimple(fnArgs)
        if ll < best:
            best = ll
            bestParamsSimple = fnArgs

    # with Pool(2) as p:
    # results = []
    start = time.time()
    fit = minimizerr(costFnSimple, bestParamsSimple, {
                            "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})
    pf = fit["x"]
    print("took", time.time() - start)
    print(fit)
    best_ll = fit["fun"]

    paramsForLevel2 = tensor( [ *pf, (pf[-1] + pf[-2]) / 2 ] )
    print("paramsForLevel2", paramsForLevel2)

    start = time.time()
    fit2 = minimizerr(costFn, paramsForLevel2, {
                            "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})

    print("took", time.time() - start)
    best_ll2 = fit2["fun"]
    pf2 = fit["x"]
    print(fit2)

    return {"lls": fit2["fun"], "llsAll": [fit["fun"], fit2["fun"]], "params": fit2["x"], "paramsAll": [fit["x"], fit2["x"]], "trajectoryLLs": trajectoryLLs, "trajectoryPi": trajectoryPis, "trajectoryAlphas": trajectoryAlphas}

def fitFnBivariateStackedDirichlet(altCountsByGene, pDs, K=4, debug=False, **kwargs):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    trajectoryPisSimple = []
    trajectoryAlphasSimple = []
    trajectoryLLsSimple = []
    _, _, _, _, costFnDirichlet, costFnSimpleDirichlet = likelihoodBivariateFast(
        altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)

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
    pisDist = Uniform(100, 25000)
    fitSimple = None


    best = float("inf")
    bestParamsSimple = []
    for y in range(10):
        fnArgs = [*pisDist.sample([K, ]), *alphasDist.sample([K-1, ]).numpy()]
        print("fnArgs", fnArgs)
        ll = costFnSimpleDirichlet(fnArgs)
        if ll < best:
            best = ll
            bestParamsSimple = fnArgs
    print("best params are", bestParamsSimple)
    # with Pool(2) as p:
    # results = []
    start = time.time()
    fit = minimizerr(costFnSimpleDirichlet, bestParamsSimple, {
                            "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})
    pf = fit["x"]
    print("took", time.time() - start)
    print(fit)
    inferredPis = Dirichlet(tensor(pf[0:4])).mean
    print("inferred Pis", inferredPis)
    best_ll = fit["fun"]

    paramsForLevel2 = tensor( [ *pf, (pf[-1] + pf[-2]) / 2 ] )
    print("paramsForLevel2", paramsForLevel2)

    start = time.time()
    fit2 = minimizerr(costFnDirichlet, paramsForLevel2, {
                            "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})

    print("took", time.time() - start)
    best_ll2 = fit2["fun"]
    pf2 = fit["x"]
    print(fit2)
    inferredPis = Dirichlet(tensor(pf2[0:4])).mean
    print("inferred Pis", inferredPis)

    return {"lls": fit2["fun"], "llsAll": [fit["fun"], fit2["fun"]], "params": fit2["x"], "paramsAll": [fit["x"], fit2["x"]], "trajectoryLLs": trajectoryLLs, "trajectoryPi": trajectoryPis, "trajectoryAlphas": trajectoryAlphas}

def fitFnBivariateFull(altCountsByGene, pDs, K=4, debug=False):
    trajectoryPis = []
    trajectoryAlphas = []
    trajectoryLLs = []
    trajectoryPisSimple = []
    trajectoryAlphasSimple = []
    trajectoryLLsSimple = []
    costFn, costFnSimple, costFnBoth, costFn2 = likelihoodBivariateFast(
        altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)

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
    fitSimple = None


    best = float("inf")
    bestParams = []
    for y in range(10):
        pi0 = pi0Dist.sample()
        pis = Uniform(1/nGenes, 1-pi0).sample([K])
        pis = pis/(pis.sum() + pi0)
        fnArgs = [*pis.numpy(), *alphasDist.sample([K, ]).numpy()]

        ll = costFnBoth(fnArgs)
        if ll < best:
            best = ll
            bestParams = fnArgs
    start = time.time()
    fit = minimizerr(costFnBoth, bestParams, {
                            "method": "Nelder-Mead", "options": {"maxiter": 20000, "adaptive": True}})

    print("took", time.time() - start)
    print("fit", fit)
    fit = fit

    return {"lls": fit["fun"], "llsAll": [fit["fun"]], "params": fit["x"], "paramsAll": [fit["x"]], "trajectoryLLs": trajectoryLLs, "trajectoryPi": trajectoryPis, "trajectoryAlphas": trajectoryAlphas}


def initBetaParams(mu, variance):
    alpha = ((1 - mu) / variance - 1 / variance) * mu**2
    beta = alpha * (1/mu - 1)

    return alpha, beta

# this doesn't appear to work well
# alphas fit for pis .05, .05, .05 were tensor([8.6943e-01, 1.4185e-30, 2.8410e-02, 1.0216e-01], dtype=torch.float64) for example (correct for pi0, not for others)
# and epochs times 4x longer (1800s)
# def likelihoodBivariateFastDirichletPrior(altCountsFlat, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs):
#     print(altCountsFlat.shape)

#     # TODO: make this flexible for multivariate
#     nHypotheses = 4
#     likelihoodFn, allNull2, likelihoodFnNoLatent = effectLikelihood(nHypotheses, pDs, altCountsFlat)
#     def jointLikelihood(params):
#         piAlpha0, piAlpha1, piAlpha2, piAlphaBoth, alpha0, alpha1, alpha2, alphaBoth = params

#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or piAlpha0 < 0 or piAlpha1 < 0 or piAlpha2 < 0 or piAlphaBoth < 0:
#             return float("inf")

#         pis = Dirichlet(concentration=tensor([piAlpha0, piAlpha1, piAlpha2, piAlphaBoth])).sample()

#         h0 = pis[0] * allNull2

#         trajectoryPis.append(pis)
#         trajectoryAlphas.append([alpha0, alpha1, alpha2, alphaBoth])

#         hs = pis[1:] * likelihoodFn(alpha0, alpha1, alpha2, alphaBoth)

#         ll = -torch.log(h0 + hs.sum(1)).sum()
#         trajectoryLLs.append(ll)
#         return ll
#     return jointLikelihood

# def fitFnBivariateMTDirichletPrior(altCountsFlat, pDs, nEpochs=20, minLLThresholdCount=100, K=4, debug=False, costFnIdx=0, method="nelder-mead"):
#     args = [altCountsFlat, pDs, 1, minLLThresholdCount,
#             K, debug, costFnIdx, method]

#     results = []

#     with Pool(cpu_count()) as p:
#         processors = []
#         for i in range(nEpochs):
#             processors.append(p.apply_async(
#                 processor, (i, *args), callback=lambda res: results.append(res)))
#         # Wait for the asynchrounous reader threads to finish
#         [r.get() for r in processors]
#         print("Got results")

#         print(results)
#         return results

# def fitFnBivariateDirichletPrior(altCountsByGene, pDs, nEpochs=20, minLLThresholdCount=100, K=4, debug=False, costFnIdx=0, method="nelder-mead"):
#     trajectoryPis = []
#     trajectoryAlphas = []
#     trajectoryLLs = []
#     costFn = likelihoodBivariateFastDirichletPrior(altCountsByGene, pDs, trajectoryPis, trajectoryAlphas, trajectoryLLs)
#     print("method", method, "costFn", costFn)

#     assert(method == "nelder-mead" or method ==
#            "annealing" or method == "basinhopping")

#     llsAll = []
#     lls = []
#     params = []

#     minLLDiff = 1
#     thresholdHitCount = 0

#     nGenes = len(altCountsByGene)

#     # pDgivenV can't be smaller than this assuming allele freq > 1e-6 and rr < 100
#     # P(V|D) * P(D) / P(V)
#     pi0Dist = Uniform(.5, 1)
#     alphasDist = Uniform(100, 25000)

#     for i in range(nEpochs):
#         start = time.time()

#         if method == "nelder-mead" or method == "basinhopping":
#             best = float("inf")
#             bestParams = []
#             for y in range(200):
#                 pi0 = pi0Dist.sample()
#                 pis = Uniform(1/nGenes, 1).sample([K])
#                 pis = pis/(pis.sum())
#                 fnArgs = [*pis.numpy(), *alphasDist.sample([K, ]).numpy()]

#                 ll = costFn(fnArgs)
#                 if ll < best:
#                     best = ll
#                     bestParams = fnArgs

#             print(f"best ll: {best}, bestParams: {bestParams}")

#             if method == "nelder-mead":
#                 fit = scipy.optimize.minimize(
#                     costFn, x0=bestParams, method='Nelder-Mead', options={"maxiter": 20000, "adaptive": True})
#             elif method == "basinhopping":
#                 fit = scipy.optimize.basinhopping(costFn, x0=bestParams)
#             else:
#                 raise Exception("should have been nelder-mead or basinhopping")
#         elif method == "annealing":
#             fit = scipy.optimize.dual_annealing(costFn, [(
#                 0.001, .999), (.001, .999), (.001, .999), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000), (100, 1_000_000_000)])

#         print("Epoch took", time.time() - start)

#         if debug:
#             print(f"epoch {i}")
#             print(fit)

#         if not fit["success"] is True:
#             if debug:
#                 print("Failed to converge")
#                 print(fit)
#             continue

#         pi1, pi2, piBoth, alpha0, alpha1, alpha2, alphaBoth = fit["x"]
#         if alpha0 < 0 or alpha1 < 0 or alpha2 < 0 or alphaBoth < 0 or pi1 < 0 or pi1 > 1 or pi2 < 0 or pi2 > 1 or piBoth < 0 or piBoth > 1:
#             if debug:
#                 print("Failed to converge")
#                 print(fit)
#             continue

#         ll = fit["fun"]
#         llsAll.append(ll)
#         if len(lls) == 0:
#             lls.append(ll)
#             params.append(fit["x"])
#             continue

#         minPrevious = min(lls)

#         if debug:
#             print("minPrevious", minPrevious)

#         # TODO: take mode of some pc-based cluster of parameters, or some auto-encoded cluster
#         if ll < minPrevious and (minPrevious - ll) >= minLLDiff:
#             if debug:
#                 print(f"better by at >= {minLLDiff}; new ll: {fit}")

#             lls.append(ll)
#             params.append(fit["x"])

#             thresholdHitCount = 0
#             continue

#         thresholdHitCount += 1

#         if thresholdHitCount == minLLThresholdCount:
#             break

#     return {"lls": lls, "llsAll": llsAll, "params": params, "trajectoryLLs": trajectoryLLs, "trajectoryPi": trajectoryPis, "trajectoryAlphas": trajectoryAlphas}
