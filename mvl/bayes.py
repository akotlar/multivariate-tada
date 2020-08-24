from .likelihoods import nullLikelihood, effectLikelihood, getAlphas, getPis
from .genData import genAlleleCount
from torch.distributions import Binomial
from torch import tensor, randperm, sort, cumsum
import torch
from matplotlib import pyplot

# P(X|HA) = P(X|h1)*P(H1) + P(X|H2)*P(H2) + P(X|HBoth)*P(HBoth)
# def bayesFactor(altCounts, sampleSizes, )

# Generates a bunch of null counts for a single gene at some allele frequency,
# then calculates the bayes factors given some alphas that were trained
# on a non-null dataset
# Gives us approximate average false positive rate, or p-value
# TADA's program has a "counts" variable for genes, but ours don't need that,
# the marginal counts for nulls are proportional to the prevalence
# def bfNullGene(alphas, nCases = tensor([1e4, 1e4, 4e3]), nCtrls = tensor(1e5), af = 1e-4, pDs = None, nIterations = tensor([10_000]), ):
#     totalSamples = nCases.sum() + nCtrls

#     if pDs is None:
#         pDs = nCases / (nCases.sum() + nCtrls)

#     rrs =  tensor([1.,1.,1.])
#     altCounts, p = genAlleleCount(totalSamples = totalSamples, rrs = rrs, af = 1e-4, pDs = pDs, sampleShape = nIterations, approx = True)
#     print(altCounts.shape)

#     # print("data",altCounts)
#     bf = bayesFactors(altCounts, pDs, alphas)
#     print("bf", bf)
#     return bf, altCounts

# this gives P(H0)/P(H1), rather that bfNullGene P(H1)/P(H0) above

def genNullData(alphas, pis, nCases, nCtrls, afMean=1e-4, pDs=None, nIterations=tensor([20_000]), **kwargs):
    print("Params:", alphas, pis, nCases, nCtrls, afMean, pDs, nIterations)
    totalSamples = nCases.sum() + nCtrls

    if pDs is None:
        pDs = nCases / (nCases.sum() + nCtrls)

    rrs = tensor([1., 1., 1.])
    altCounts, p = genAlleleCount(
        totalSamples=totalSamples, rrs=rrs, afMean=afMean, pDs=pDs, sampleShape=nIterations, approx=True)
    return altCounts, p

def bfNullGenePosterior(alphas, pis, nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(1e5), afMean=1e-4, pDs=None, nIterations=tensor([10_000]), **kwargs):
    altCounts, _ = genNullData(alphas, pis, nCases, nCtrls, afMean, pDs, nIterations)

    bf = bfdp(altCounts, pDs, alphas, pis)
    # print(altCounts.shape)
    # print("bf", bf)
    return bf, altCounts


def bfNullGenePosteriorFlatPrior(alphas, nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(1e5), afMean=1e-4, pDs=None, nIterations=tensor([10_000]), nAlternateHypotheses=3, **kwargs):
    nHypotheses = nAlternateHypotheses + 1
    pis = tensor(1 / nHypotheses).expand([nAlternateHypotheses, ])
    return bfNullGenePosterior(alphas, pis, nCases, nCtrls, afMean, pDs, nIterations)

# Doesn't seem to produce desired results...at least with altCounts that are largely 0's..
# def permutedGeneCountBFs(altCounts, alphas, pDs):
#     # https://discuss.pytorch.org/t/shuffling-a-tensor/25422/4
#     idx = randperm(altCounts.nelement())
#     print('idx', idx[0:2000].numpy())
#     altCountsPermuted = altCounts.view(-1)[idx].view(altCounts.size())
#     print("altCountsPermuted", altCountsPermuted)

#     bf = bayesFactors(altCountsPermuted, pDs, alphas)
#     print("bf", bf)
#     return bf, altCountsPermuted

# def permutedGeneCountBFsHA(altCounts, alphas, pDs):
#     # https://discuss.pytorch.org/t/shuffling-a-tensor/25422/4
#     idx = randperm(altCounts.nelement())
#     print('idx', idx[0:2000].numpy())
#     altCountsPermuted = altCounts.view(-1)[idx].view(altCounts.size())
#     print("altCountsPermuted", altCountsPermuted)

#     bf = bayesFactorsHA(altCountsPermuted, pDs, alphas)
#     print("bf", bf)
#     return bf, altCountsPermuted

# def bayesFactors(altCounts, pDs, alphas):
#     likelihoodFn, nullLikes = effectLikelihood(4, pDs, altCounts)

#     effectLikes = likelihoodFn(*alphas)
#     nullLikes = nullLikelihood(tensor([1 - pDs.sum(), *pDs]), altCounts).expand(effectLikes.T.shape).T

#     return effectLikes / nullLikes

# def bayesFactorsHA(altCounts, pDs, alphas):
    # likelihoodFn, nullLikes = effectLikelihood(4, pDs, altCounts)

    # effectLikes = likelihoodFn(*alphas)
    # nullLikes = nullLikelihood(tensor([1 - pDs.sum(), *pDs]), altCounts)
    # # In reality we would take the maximum likelihood of effectLikes, and compare it to nullLikes
    # # since apriori we wouldn't know which to choose
    # print(effectLikes.max(1))
    # return effectLikes.max(1).values / nullLikes

# same as TADA's
# TODO: OK, alternate thought. Maybe we multiply all of the bayes factors together
# and then check the FDR of that?
# def bayesFDR(bf, pi):
#     print("bf", bf)

#     assert(len(bf.shape) == 1) #vector

#     bfSorted, originalIndices = sort(bf, descending=True)

#     nonInfIndices = torch.nonzero(bfSorted != float("inf"))
#     # TODO: -nonInfIndices doesn't work, investigate
#     bfSorted[torch.nonzero(bfSorted == float("inf"))] = bfSorted[nonInfIndices].max()

#     q = (pi * bfSorted) / (1 - pi + pi * bfSorted)
#     qNull = 1 - q # posterior probability of null model

#     fdr = cumsum(qNull, dim=0)/tensor(list(range(1, len(bfSorted) + 1)))

#     return fdr[originalIndices]

# ROC AUC curve for classifying a risk gene as a risk gene
# (rather than belonging to any one class)

def aucROC(fit, input, params, name):
    bfdpNull, altCountsNullBfPerm = bfNullGenePosterior(getAlphas(fit), getPis(fit), nIterations=tensor([50_000]), **params)
    bfdpData = bfdp(altCounts=input["altCounts"], pDs=params["pDs"], alphas=tensor(fit["params"][0][3:]), pis=tensor(fit["params"][0][0:3]), )

    genAucRocRiskGene(bfdpNull, bfdpData, affectedGenes1=tensor(input["affectedGenes"][0]), affectedGenes2=tensor(input["affectedGenes"][1]), affectedGenesBoth=tensor(input["affectedGenes"][2]), name=name)

def aucROCflatPrior(fit, input, params, name):
    bfdpNull, altCountsNullBfPerm = bfNullGenePosteriorFlatPrior(getAlphas(fit), nIterations=tensor([50_000]), **params)
    bfdpData = bfdpFlatPrior(altCounts=input["altCounts"], pDs=params["pDs"], alphas=tensor(fit["params"][0][3:]))

    genAucRocRiskGene(bfdpNull, bfdpData, affectedGenes1=tensor(input["affectedGenes"][0]), affectedGenes2=tensor(input["affectedGenes"][1]), affectedGenesBoth=tensor(input["affectedGenes"][2]), name=name)

def genAucRocRiskGene(bfdpNull, bfdpData, affectedGenes1, affectedGenes2, affectedGenesBoth, name):
    from sklearn.metrics import auc
    # Precision recall curve generator
    ###### TODO: place into bayes.py ######
    bfdpCutoffs = []
    # 1/specificity aka % wrong
    specificities = []

    # recall
    sensitivity1 = []
    sensitivity2 = []
    sensitivityBoth = []
    sensitivityAll = []
    recallBoth = []
    recallAllAffected = []
    recall = []

    # # torch.cumsum(bfdpNull.sort().values, 0)
    # bfNeeded = 0
    # total = len(bfdpNullHAvsH0)
    # for bf in range(1, 500000, 1):
    #     bfTest = bf / 50000
    #     mask = bfdpNullHAvsH0 > bfTest
    # #     print(mask)
    #     wrong = len(torch.nonzero(mask)) / total
    # #     print("at", bfTest, "total", wrong)
    #     if wrong <= .1:
    #         print("empirical p", bfTest, wrong)
    #         bfNeeded = bfTest
    #         break

    # bfNeededH0vsHA = 1 / bfNeeded
    # print(bfNeededH0vsHA, bfNeeded)
    bfdpNullHAvsH0 = 1 / bfdpNull  # bfdpNull is H0 / H1
    allAffectedGenes = torch.cat(
        [affectedGenes1, affectedGenes2, affectedGenesBoth])
    for bf in range(10_000, 100, -1):
        bfTest = bf / 100

    #     calledAlternate = bfdpNullHAvsH0 > bfTest #wrong, called HA when H0 true
        calledNull = bfdpNullHAvsH0 <= bfTest  # wrong, called HA when H0 true

        # call negative / all negatives, aka true negative rate
        specificity = len(torch.nonzero(calledNull)) / len(bfdpNullHAvsH0)

        specificities.append(specificity)

        inferredRisk = bfdpData <= 1 / bfTest  # bfdp is P(H0) / P(H1)

        nGenes1 = len(affectedGenes1)
        nGenes1AssumedHA = len(torch.nonzero(inferredRisk[affectedGenes1]))
        power1 = nGenes1AssumedHA / nGenes1

        sensitivity1.append(power1)

        nGenes2 = len(affectedGenes2)
        nGenes2AssumedHA = len(torch.nonzero(inferredRisk[affectedGenes2]))
        power2 = nGenes2AssumedHA / nGenes2

        sensitivity2.append(power2)

        nGenesBoth = len(affectedGenesBoth)
        nGenesBothAssumedHA = len(torch.nonzero(
            inferredRisk[affectedGenesBoth]))
        powerBoth = nGenesBothAssumedHA / nGenesBoth

        sensitivityBoth.append(powerBoth)

        nGenesAll = len(allAffectedGenes)
        nGenesAllAssumedHA = len(torch.nonzero(inferredRisk[allAffectedGenes]))
        powerAcrossAll = nGenesAllAssumedHA / nGenesAll

        sensitivityAll.append(powerAcrossAll)

        # bfNeededH0vsHA = 1 /bfNeeded
#         print("testing", bfTest)
#         print("specificity", specificity)
#         print(len(inferredRisk[affectedGenes1Upscale]))
#         print("power1", power1)
#         print("power2", power2)
#         print("powerBoth", powerBoth)
#         print("powerAcrossAll", powerAcrossAll)
#         print(bfNeededH0vsHA)
    if specificities[-1] > 0:
        specificities.append(0)
        sensitivityAll.append(sensitivityAll[-1])
        sensitivity2.append(sensitivity2[-1])
        sensitivity1.append(sensitivity1[-1])
        sensitivityBoth.append(sensitivityBoth[-1])

        print("Added specificity 0")

    fpr = 1 - tensor(specificities)

    # print("worst fpr:", fpr.max())
    aucScoreAll = "{:f}".format(auc(fpr, sensitivityAll))
    aucScore1 = "{:f}".format(auc(fpr, sensitivity1))
    aucScore2 = "{:f}".format(auc(fpr, sensitivity2))
    aucScoreBoth = "{:f}".format(auc(fpr, sensitivityBoth))

    pyplot.plot(fpr, sensitivityAll)
    pyplot.plot(fpr, sensitivity1)
    pyplot.plot(fpr, sensitivity2)
    pyplot.plot(fpr, sensitivityBoth)

    pyplot.legend([f"Overall AUC={aucScoreAll}", f"Gene1 AUC={aucScore1}",
                   f"Gene2 AUC={aucScore2}", f"GeneBoth AUC={aucScoreBoth}"])

    pyplot.savefig(name)


# ROC AUC curve for classifying a risk gene as a risk gene
# (rather than belonging to any one class)
def genAucRocSingleGene(bfdpNull, affectedGenes1, affectedGenes2, affectedGenesBoth, name):
    from sklearn.metrics import auc
    # Precision recall curve generator
    ###### TODO: place into bayes.py ######
    bfdpCutoffs = []
    # 1/specificity aka % wrong
    specificities = []

    # recall
    sensitivity1 = []
    sensitivity2 = []
    sensitivityBoth = []
    sensitivityAll = []
    recallBoth = []
    recallAllAffected = []
    recall = []

    # bfNeededH0vsHA = 1 / bfNeeded
    # print(bfNeededH0vsHA, bfNeeded)
    bfdpNullHAvsH0 = 1 / bfdpNull  # bfdpNull is H0 / H1
    allAffectedGenes = torch.cat(
        [affectedGenes1, affectedGenes2, affectedGenesBoth])
    for bf in range(10_000, 100, -1):
        bfTest = bf / 100

    #     calledAlternate = bfdpNullHAvsH0 > bfTest #wrong, called HA when H0 true
        calledNull = bfdpNullHAvsH0 <= bfTest  # wrong, called HA when H0 true

        # call negative / all negatives, aka true negative rate
        specificity = len(torch.nonzero(calledNull)) / len(bfdpNullHAvsH0)

        specificities.append(specificity)

        inferredRisk = bfdpData <= 1 / bfTest  # bfdp is P(H0) / P(H1)

        nGenes1 = len(affectedGenes1)
        nGenes1AssumedHA = len(torch.nonzero(inferredRisk[affectedGenes1]))
        power1 = nGenes1AssumedHA / nGenes1

        sensitivity1.append(power1)

        nGenes2 = len(affectedGenes2)
        nGenes2AssumedHA = len(torch.nonzero(inferredRisk[affectedGenes2]))
        power2 = nGenes2AssumedHA / nGenes2

        sensitivity2.append(power2)

        nGenesBoth = len(affectedGenesBoth)
        nGenesBothAssumedHA = len(torch.nonzero(
            inferredRisk[affectedGenesBoth]))
        powerBoth = nGenesBothAssumedHA / nGenesBoth

        sensitivityBoth.append(powerBoth)

        nGenesAll = len(allAffectedGenes)
        nGenesAllAssumedHA = len(torch.nonzero(inferredRisk[allAffectedGenes]))
        powerAcrossAll = nGenesAllAssumedHA / nGenesAll

        sensitivityAll.append(powerAcrossAll)

        # bfNeededH0vsHA = 1 /bfNeeded
#         print("testing", bfTest)
#         print("specificity", specificity)
#         print(len(inferredRisk[affectedGenes1Upscale]))
#         print("power1", power1)
#         print("power2", power2)
#         print("powerBoth", powerBoth)
#         print("powerAcrossAll", powerAcrossAll)
#         print(bfNeededH0vsHA)
    specificities = tensor(specificities)

    aucScoreAll = "{:f}".format(auc(1 - specificities, sensitivityAll))
    aucScore1 = "{:f}".format(auc(1 - specificities, sensitivity1))
    aucScore2 = "{:f}".format(auc(1 - specificities, sensitivity2))
    aucScoreBoth = "{:f}".format(auc(1 - specificities, sensitivityBoth))

    pyplot.plot(1 - specificities, sensitivityAll)
    pyplot.plot(1 - specificities, sensitivity1)
    pyplot.plot(1 - specificities, sensitivity2)
    pyplot.plot(1 - specificities, sensitivityBoth)

    pyplot.legend([f"Overall AUC={aucScoreAll}", f"Gene1 AUC={aucScore1}",
                   f"Gene2 AUC={aucScore2}", f"GeneBoth AUC={aucScoreBoth}"])

    pyplot.savefig(name)


def bfdpThreshold(alphas, pis, nCases, nCtrls, afMean, pDs, targetFDR=.1, nIterations=tensor([50_000])):
    bfdpNull, _ = bfNullGenePosterior(
        alphas, pis, nCases=nCases, nCtrls=nCtrls, af=afMean, pDs=pDs, nIterations=tensor([50_000]))

    bfNeeded = 0
    bfdpNullHAvsH0 = 1 / bfdpNull
    total = len(bfdpNullHAvsH0)
    divisor = 50_000
    targetFDR = .1
    fdr = 0
    seen = False
    for bf in range(int(divisor / 2), divisor * 100, 1):
        bfTest = bf / divisor
        mask = bfdpNullHAvsH0 > bfTest

        fdr = len(torch.nonzero(mask)) / total

        if fdr <= targetFDR:
            print("empirical fdr", bfTest, fdr)
            bfNeeded = bfTest
            break

    bfNeededH0vsHA = 1 / bfNeeded
    return bfNeededH0vsHA, fdr


def bfdpThreshold2(bfdpNull, alphas, pis, nCases, nCtrls, afMean, pDs, targetFDR=.1, nIterations=tensor([50_000]), **kwargs):
    bfNeeded = 0
    bfdpNullHAvsH0 = 1 / bfdpNull
    total = len(bfdpNullHAvsH0)
    divisor = 50_000
    targetFDR = .1
    fdr = 0
    seen = False
    for bf in range(int(divisor / 2), divisor * 100, 1):
        bfTest = bf / divisor
        mask = bfdpNullHAvsH0 > bfTest

        fdr = len(torch.nonzero(mask)) / total

        if fdr <= targetFDR:
            print("empirical fdr", bfTest, fdr)
            bfNeeded = bfTest
            break

    return 1/bfNeeded, fdr

# https://projecteuclid.org/download/pdf_1/euclid.lnms/1215540968


def bfdpThresholdFlatPrior(alphas, nCases, nCtrls, afMean, pDs, targetFDR=.1, nIterations=tensor([50_000]), nAlternateHypotheses=3):
    nHypotheses = nAlternateHypotheses + 1
    pis = tensor(1 / nHypotheses).expand([nAlternateHypotheses, ])
    return bfdpThreshold(alphas, pis, nCases, nCtrls, afMean, pDs, targetFDR=targetFDR, nIterations=nIterations)


# Calculate P(H0|data)/sum(P(H|data))
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1950810/
# bfdp = Pr(H_0|y) = p(y|theta_h0, h_0)*p(H0) / sum(over all hypothesis) { p(y|theta_that_hypothesis, that_hypothesis)*pi_0 + p(y|h1)*pi_1 }
def bfdp(altCounts, pDs, alphas, pis, nAlternateHypotheses=3):
    assert(nAlternateHypotheses == 3)
    likelihoodFn, nullLikes, likelihoodFnSimpleNoLatent = effectLikelihood(
        nAlternateHypotheses + 1, pDs=pDs, altCountsFlat=altCounts)

    effectLikes = likelihoodFn(*alphas)
    nullLikes = nullLikelihood(tensor([1 - pDs.sum(), *pDs]), altCounts)

    piNull = 1 - pis.sum()

    numerator = piNull * nullLikes

    denominator = pis * effectLikes

    res = numerator / (numerator + denominator.sum(1))

    # print("alphas", alphas)
    # print("nullLikes", nullLikes)
    # print("effectLikes", effectLikes)
    # print("altCounts", altCounts)
    # print("numerator", numerator)
    # print("pis", pis)
    # print("denominator", denominator)
    # print("denominator/.sum(1)", denominator.sum(1))
    # print("res", res)

    return res

def bfdpOneAltHypothesis(altCounts, pDs, alphas, pis, nAlternateHypotheses=3, hIdx=0):
    print("altCounts", altCounts.size())
    assert(nAlternateHypotheses == 3)
    likelihoodFn, nullLikes, likelihoodFnSimpleNoLatent = effectLikelihood(
        nAlternateHypotheses + 1, pDs=pDs, altCountsFlat=altCounts)

    effectLikes = likelihoodFn(*alphas)
    nullLikes = nullLikelihood(tensor([1 - pDs.sum(), *pDs]), altCounts)
    print("effectLikes", effectLikes.shape)
    piNull = 1 - pis.sum()

    null = piNull * nullLikes

    alt = pis * effectLikes

    res = alt[:, hIdx] / (null + alt.sum(1))

    return res


def bfdpFlatPrior(altCounts, pDs, alphas, nAlternateHypotheses=3):
    assert(nAlternateHypotheses == 3)
    nHypotheses = nAlternateHypotheses + 1
    pis = tensor(1 / nHypotheses).expand([nAlternateHypotheses, ])
    return bfdp(altCounts, pDs, alphas, pis, nAlternateHypotheses)

#  Calculates P(H_N|Data) / sum(P(H|data))


def bfdpAlternates(altCounts, pDs, alphas, pis, nAlternateHypotheses=3):
    assert(nAlternateHypotheses == 3)

    likelihoodFn, nullLikes, likelihoodFnSimpleNoLatent = effectLikelihood(
        nAlternateHypotheses + 1, pDs=pDs, altCountsFlat=altCounts)

    effectLikes = likelihoodFn(*alphas)
    nullLikes = nullLikelihood(tensor([1 - pDs.sum(), *pDs]), altCounts)

    piNull = 1 - pis.sum()
    null = piNull * nullLikes
    effects = pis * effectLikes
    denom = null + effects.sum(1)

    print("effects", effects, "\neffects[:, 0].shape: ", effects[:, 0].shape)
    print("denom", denom, "\ndenom.shape:", denom.shape)

    h0 = null / denom
    h1 = effects[:, 0] / denom
    h2 = effects[:, 1] / denom
    h3 = effects[:, 2] / denom

    print("h0, h1, h2, h3\n", h0, "\n", h1, "\n", h2, "\n", h3)
    print("effects[:, 0] / null", effects[:, 0] / null)

    return h0, h1, h2, h3


def bfdpAlternatesFlatPrior(altCounts, pDs, alphas, nAlternateHypotheses=3):
    assert(nAlternateHypotheses == 3)
    nHypotheses = nAlternateHypotheses + 1
    pis = tensor(1 / nHypotheses).expand([nAlternateHypotheses, ])

    return bfdpAlternates(altCounts=altCounts, pDs=pDs, alphas=alphas, pis=pis, nAlternateHypotheses=nAlternateHypotheses)

# For now, framing FDR on H0: non-risk H1: risk gene, vs the more granular consideration
# TODO: do this for a single hypothesis or all?
# TODO: not sure if this  is entirely right, because piNull needs to reflect all possible hypothesis, not just 1 - pi
# where pi is probability of one of the mixture components

# OK fuck it, for now will do for a single hypothesis at a time


def bayesFDRs(bayesFactors, pis):
    assert(bayesFactors.shape[1] == len(pis))  # 3 hypotheses, for now

    return tensor([bayesFDR(bayesFactors[:, i], pis[i]).numpy() for i in range(len(pis))])


def bayesFDRsHA(bayesFactors, pis):
    assert(bayesFactors.shape[1] == len(pis))  # 3 hypotheses, for now

    return tensor(bayesFDR(bayesFactors.max(1).values, pis.sum()))

# oh, pi0 here is our P(H0)
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
