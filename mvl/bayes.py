from .likelihoods import nullLikelihood, effectLikelihood
from .genData import genAlleleCount
from torch.distributions import Binomial
from torch import tensor, randperm, sort, cumsum
import torch
# def bayesFactor(altCounts, sampleSizes, )

# Generates a bunch of null counts for a single gene at some allele frequency,
# then calculates the bayes factors given some alphas that were trained
# on a non-null dataset
# Gives us approximate average false positive rate, or p-value
# TADA's program has a "counts" variable for genes, but ours don't need that,
# the marginal counts for nulls are proportional to the prevalence
def bfNullGene(alphas, nCases = tensor([1e4, 1e4, 4e3]), nCtrls = tensor(1e5), af = 1e-4, pDs = None, nIterations = tensor([10_000]), ):
    totalSamples = nCases.sum() + nCtrls

    if pDs is None:
        pDs = nCases / (nCases.sum() + nCtrls)

    rrs =  tensor([1.,1.,1.])
    altCounts, p = genAlleleCount(totalSamples = totalSamples, rrs = rrs, af = 1e-4, pDs = pDs, nToSample = nIterations, approx = True)
    print(altCounts.shape)
    
    # print("data",altCounts)
    bf = bayesFactors(altCounts, pDs, alphas)
    print("bf", bf)
    return bf, altCounts

# Doesn't seem to produce desired results...at least with altCounts that are largely 0's..
def permutedGeneCountBFs(altCounts, alphas, pDs):
    # https://discuss.pytorch.org/t/shuffling-a-tensor/25422/4
    idx = randperm(altCounts.nelement())
    print('idx', idx[0:2000].numpy())
    altCountsPermuted = altCounts.view(-1)[idx].view(altCounts.size())
    print("altCountsPermuted", altCountsPermuted)
    
    bf = bayesFactors(altCountsPermuted, pDs, alphas)
    print("bf", bf)
    return bf, altCountsPermuted
    
def bayesFactors(altCounts, pDs, alphas):
    likelihoodFn, nullLikes = effectLikelihood(4, pDs, altCounts)

    effectLikes = likelihoodFn(*alphas)
    nullLikes = nullLikelihood(tensor([1 - pDs.sum(), *pDs]), altCounts).expand(effectLikes.T.shape).T
    
    return effectLikes / nullLikes

# same as TADA's
def bayesFDR(bf, pi):
    piNull = 1 - pi

    assert(len(bf.shape) == 1) #vector

    bfSorted, originalIndices = sort(bf, descending=True)

    nonInfIndices = torch.nonzero(bfSorted != float("inf"))
    # TODO: -nonInfIndices doesn't work, investigate
    bfSorted[torch.nonzero(bfSorted == float("inf"))] = bfSorted[nonInfIndices].max()

    q = (pi * bfSorted) / (piNull + pi * bfSorted)
    qNull = 1 - q # posterior probability of null model

    fdr = cumsum(qNull, dim=0)/tensor(list(range(1, len(bfSorted) + 1)))

    return fdr[originalIndices]

# For now, framing FDR on H0: non-risk H1: risk gene, vs the more granular consideration
# TODO: do this for a single hypothesis or all?
# TODO: not sure if this  is entirely right, because piNull needs to reflect all possible hypothesis, not just 1 - pi
# where pi is probability of one of the mixture components

# OK fuck it, for now will do for a single hypothesis at a time
def bayesFDRs(bayesFactors, pis):
    assert(bayesFactors.shape[1] == len(pis)) # 3 hypotheses, for now

    return tensor([bayesFDR(bayesFactors[:, i], pis[i]).numpy() for i in range(len(pis))])

# Bayesian.FDR <- function(BF, pi0) {
#   # Bayesian FDR control (PMID:19822692, Section2.3)
#   # BF: a svector of BFs 
#   # pi0: the prior probability that the null model is true
#   # Return: the q-value of each BF, and the number of findings with q below alpha.
  
#   # order the BF in decreasing order, need to retain order to get results back in proper order 
#   i.order=order(BF, decreasing = T)
#   BF=BF[i.order]
#   # convert BFs to PPA (posterior probability of alternative model)
#   pi <- 1-pi0
#   q <- pi*BF/(1-pi+pi*BF) # PPA
#   q0 <- 1 - q # posterior probability of null model
  
#   # the FDR at each PPA cutoff
#   FDR=cumsum(q0)/(1:length(BF))
  
#   # reorder to the original order
#   FDR[i.order]=FDR
  
#   return (FDR=FDR)
# }

# bayesFactor.pvalue <- function(BF,BF.null){
#   ## determines the pvalue for the BF using permutations under the null hypothesis BF.null
#   # BF : vector with bayes factors  based on the data
#   # BF.null : vector with bayes factors based on permuted data 
  
#   BF.null <- sort(BF.null, decreasing=TRUE)
#   pval <- findInterval(-BF, -BF.null)/length(BF.null)
#   pval[pval==0] <- 0.5/length(BF.null)
  
#   return(pval=pval)
# }