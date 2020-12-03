import torch
from torch import Tensor
import torch.tensor as tensor
from torch.distributions import Gamma, Categorical, Categorical, MultivariateNormal, Multinomial, Normal as N
import numpy as np
from pyper import *
from torch.distributions import MultivariateNormal
from torch import tensor, Tensor
from scipy.stats import multivariate_normal as scimvn
import numpy as np
import torch
from typing import Tuple
    
class WrappedMVN():
    def __init__(self, mvn: MultivariateNormal):
        self.mvn = mvn
        self.scimvn = scimvn(mean=self.mvn.mean, cov=self.mvn.covariance_matrix)

    def cdf(self, lower: Tensor):
        l = lower.expand(self.mvn.mean.shape)
        return self.scimvn.cdf(l)

def genAlleleCountFromPVDS(nCases: Tensor, nCtrls: Tensor, PVDs = tensor([1.,1.,1.]), gene_af = 1e-4, pDs = tensor([.01,.01,.002]), **kwargs):
    """
    Starting from the true population estimate, P(V|D) we generate the in-sample P(D|V), and use that as our multinomial allele frequecny
    This value is approximately rr*P(D)
    We cannot simply multiply P(V|D) * P(D_hat) because the result may be larger than P(V)
    Instead we need to normalize by the difference between P(D_hat) and P(D)
    P(V|D) * P(D_hat) * P(D) / P(D_hat)? No, P(D|V) is exclusive of P(D)
    It is only later, in inference that we need to re-scale

    Generates 1 pooled control population
    """
    N = nCases.sum() + nCtrls
    PDhat = nCases / N

    PND = 1.0 - pDs.sum()
    PNDhat = 1.0 - PDhat.sum()

    pop_estimate_pvd_pd = (PVDs * pDs)
    PVND_PND_POP = gene_af - pop_estimate_pvd_pd.sum()
    assert PVND_PND_POP > 0

    PVND = PVND_PND_POP / PND

    marginalAltCount = int(torch.ceil(PVND * nCtrls + (PVDs * nCases).sum()))
    p = tensor([PVND, *PVDs]) * tensor([PNDhat, *PDhat])

    return Multinomial(probs=p, total_count=marginalAltCount).sample(), p, PVND, PVDs

def genParams(pis=tensor([.1, .1, .05]), rrShape=tensor(10.), rrMeans=tensor([3., 3., 1.5]), afShape=tensor(10.), afMean=tensor(1e-4), nCases=tensor([5e3, 5e3, 2e3]), nCtrls=tensor(5e5), covShared=tensor([[1, .5], [.5, 1]]), covSingle=tensor([[1., 0.], [0., 1.]]), meanEffectCovarianceScale=tensor(.01), pDs=None, rrtype="default", **kwargs):
    nGenes = 20_000

    assert pDs is not None

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
        "meanEffectCovarianceScale": meanEffectCovarianceScale,
        "rrtype": rrtype
    }]

def create_covariance():
    # residualVariance = totalVariance - geneticVariance
    # print("totalVariance", totalVariance, "geneticVariance", geneticVariance, "residualVariance", residualVariance)
    
    # print("phenotypicCorrelation", phenotypicCorrelation)
    # print('genetiCorrelation', geneticCorrelation)
    # print('residualCorrelation', residualCorrelation)
    # genetic_covariance = geneticCorrelation * geneticVariance
    # residual_covariance = residualCorrelation * residualVariance
    # print("genetic_covariance", genetic_covariance)
    # print("residual_covariance", residual_covariance)
    pass

# https://gist.github.com/ncullen93/58e71c4303b89e420bd8e0b0aa54bf48
def corrcoef(x):
    """
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor
    
    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.corrcoef(x)
        >>> th_corr = corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c

def liabilityParams(totalVariance = tensor([.01, .01]),
heritability = tensor([.8, .8]),
phenotypicCorrelation = tensor([[1., 5.], [.5, 1.]]),
geneticCorrelation = tensor([[1., 5.], [.5, 1.]]),
residualCorrelation = tensor([[1., 5.], [.5, 1.]])) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    geneticVariance = totalVariance * heritability
    residualVariance = totalVariance - geneticVariance

    tD = torch.diag(totalVariance) ** .5
    print("tD", tD)
    phenotypicCovariance = tD @ phenotypicCorrelation @ tD

    gD = torch.diag(geneticVariance) ** .5
    print("gD", gD)
    geneticCovariance = gD @ geneticCorrelation @ gD

    rD = torch.diag(residualVariance) ** .5

    residualCovariance = rD @ residualCorrelation @ rD

    return phenotypicCovariance, geneticCovariance, residualCovariance

# TOOD: I don't think we're handling variance correctly
# I believe if we choose to make variance < 1 we must do so in the individual normals as well
# else the thresh1 and thresh2 are too big, and PDBoth becomes tiny
def genCounts(nCases, nCtrls, pDs = tensor([.01, .01]),
    diseaseFractions = tensor([.05, .05, .01]), rrMeans = tensor([3, 5]),
    afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
    phenotypicCovariance = tensor([[1., 0.], [0., 1.]]),
    geneticCovariance = tensor([[1., 0.], [0., 1.]]),
    residualCovariance = tensor([[1., 0.], [0., 1.]]),
    fudgeFactor = tensor(.01), **kwargs):
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

    # rg = covg/torch.sqrt(hx * hy)
    ####################### Calculate P(DBoth) given genetic correlation ##############################
    phenotypicVariances = torch.diag(phenotypicCovariance)
    print(phenotypicVariances)
    n1= N(0, 1) #phenotypicVariances[0])
    thresh1 = n1.icdf(pDs[0])
    n2 = N(0, 1) #phenotypicVariances[1])
    thresh2 = n2.icdf(pDs[1])
    print("corrcoef(phenotypicCovariance)", phenotypicCovariance)
    # TODO: should this be phenotypic correlation or residual?
    # I think prevalence should be due to both due to genetic and environmental reasons
    pdBothGenerator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), phenotypicCovariance))
    PDBoth = tensor(pdBothGenerator.cdf(tensor([thresh1, thresh2])))
    pDsWithBoth = tensor([*pDs, PDBoth])

    print("pDsWithBoth", pDsWithBoth)
    ##################### Calculate effects in genes that affect both conditions #########################
    meanEffectsAcrossAllGenes = getTargetMeanEffect(pDs, rrMeans)
    print("meanEffectsAcrossAllGenes", meanEffectsAcrossAllGenes)

    effectGenerator = MultivariateNormal(meanEffectsAcrossAllGenes, geneticCovariance * fudgeFactor)
    # Shape nGenes x nIndependentEffects
    allEffects = -effectGenerator.sample([nGenes])

    # TODO: why is this sampling with variance 1?
    pd1Gen = N(allEffects[:, 0], 1)
    pd2Gen = N(allEffects[:, 1], 1)
    PD1GivenV = pd1Gen.cdf(thresh1) 
    PD2GivenV = pd2Gen.cdf(thresh2)

    PDBothGivenV = []
    for i in range(nGenes):
        mvn = MultivariateNormal(allEffects[i], torch.eye(2))
        mvnw = WrappedMVN(mvn)

        PDBothGivenV.append(mvnw.cdf(tensor([thresh1, thresh2])))
    PDBothGivenV = tensor(PDBothGivenV)
    pdvsInBoth = torch.stack([PD1GivenV, PD2GivenV, PDBothGivenV]).T

    print("allEffects", allEffects)
    print("PDBothGivenV.mean", PDBothGivenV.mean())
    print("PDBothGivenV / PDBoth", (PDBothGivenV / PDBoth).mean())
    print("pdsCovarOnMean.mean(0)", pdvsInBoth.mean(0))
    print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,1])\n", np.corrcoef(pdvsInBoth[:,0], pdvsInBoth[:,1]))
    print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,2])\n", np.corrcoef(pdvsInBoth[:,0], pdvsInBoth[:,2]))

    ############### Calculate effects in genes that affect a single conditions ##################
    effectGenerator= MultivariateNormal(meanEffectsAcrossAllGenes, residualCovariance * fudgeFactor)
    allEffectsFor12 = -effectGenerator.sample([nGenes])
    pd1Gen = N(allEffectsFor12[:, 0], 1)
    pd2Gen = N(allEffectsFor12[:, 1], 1)
    PD1Vsingle = pd1Gen.cdf(thresh1)
    PD2Vsingle = pd2Gen.cdf(thresh2)

    PDBoth1GivenV = []
    PDBoth2GivenV = []
    for i in range(nGenes):
        mvn = MultivariateNormal(tensor([allEffectsFor12[i, 0], 0]), torch.eye(2))
        mvn2 = MultivariateNormal(tensor([0, allEffectsFor12[i, 1]]), torch.eye(2))
        mvnw1 = WrappedMVN(mvn)
        mvnw2 = WrappedMVN(mvn2)

        PDBoth1GivenV.append(mvnw1.cdf(tensor([thresh1, thresh2])))
        PDBoth2GivenV.append(mvnw2.cdf(tensor([thresh1, thresh2])))
    PDBoth1GivenV = tensor(PDBoth1GivenV)
    PDBoth2GivenV = tensor(PDBoth2GivenV)

    print("PDBoth1GivenV", PDBoth1GivenV)
    print("PDBoth2GivenV", PDBoth2GivenV)

    print("np.corrcoef(PD1Vsingle, PD2Vsingle)\n", np.corrcoef(PD1Vsingle, PD2Vsingle))
    print("np.corrcoef(PD1Vsingle, PDBoth1GivenV)\n", np.corrcoef(PD1Vsingle, PDBoth1GivenV))
    print("np.corrcoef(PD2Vsingle, PDBoth1GivenV)\n", np.corrcoef(PD2Vsingle, PDBoth1GivenV))
    print("np.corrcoef(PD2Vsingle, PDBoth2GivenV)\n", np.corrcoef(PD2Vsingle, PDBoth2GivenV))

    pdvsGeneAffects1 = torch.stack([PD1Vsingle, pDs[1].expand([nGenes]), PDBoth1GivenV])
    pdvsGeneAffects2 = torch.stack([pDs[0].expand([nGenes]), PD2Vsingle, PDBoth2GivenV])
    pdvsNull = pDsWithBoth.expand(pdvsGeneAffects1.T.shape).T

    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    afs = afDist.sample([nGenes, ])
    
    print("pdvsGeneAffects1.mean", pdvsGeneAffects1.mean(0))
    print("afs.dist", afs.mean(), "+/-", afs.std())
    print("afs.shape", afs.shape)
    ############# Our multinomial probabilities are, in the margin P(V|D,gene) ###############################
    # This is decomposed into P(V|Disesase1)P(Disease1) + P(V|Disease2)P(Disease2) ... for every gene
    # To get P(V|Disease) from P(Disease|V), we note
    # P(D|V)P(V) = P(V|D)P(D), SO P(V|D) = P(D|V)*P(V) / P(D)
    # For every gene we have an allele frequency, P(V), sampled from the gamma distribution
    # Calculate penetrance P(D|V) using the mean effects for each genetic architecture
    # To get the inverse, P(V|D), use bayes theorem
    #########################################################################################################
    pvd_base = torch.stack([pdvsNull, pdvsGeneAffects1, pdvsGeneAffects2, pdvsInBoth.T]).transpose(2, 0).transpose(1,2) / pDsWithBoth
    pvds = afs.unsqueeze(-1).unsqueeze(-1).expand(pvd_base.shape) * pvd_base

    pis = tensor([1 - diseaseFractions.sum(), *diseaseFractions])
    categorySampler = Categorical(pis)
    categories  = categorySampler.sample([nGenes,])

    affectedGenes = []
    unaffectedGenes = []
    altCounts = []
    probs = []
    PVDs = []
    for geneIdx in range(nGenes):
        affects = categories[geneIdx]

        if affects == 0:
            unaffectedGenes.append(geneIdx)
        else:
            while affects - 1 >= len(affectedGenes):
                affectedGenes.append([])
            affectedGenes[affects - 1].append(geneIdx)
        altCountsGene, p, pvnd, pvd = genAlleleCountFromPVDS(nCases = nCases, nCtrls = nCtrls, PVDs = pvds[geneIdx, affects], afMean = afs[geneIdx], pDs = pDsWithBoth)

        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        PVDs.append([pvnd, *pvd])

    altCounts = tensor(altCounts)
    probs = tensor(probs)
    PVDs = tensor(PVDs)

    return {"altCounts": altCounts, "popAfs": afs, "sampleAfs": probs, "categories": categories, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "PDs": pDsWithBoth, "PVDs": PVDs}

def v6liability2(nCases, nCtrls, pDs = tensor([.01, .01]), diseaseFractions = tensor([.05, .05, .01]), rrMeans = tensor([3, 5]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
             meanEffectCovarianceScale=tensor(.01), covShared=tensor([ [1., .5], [.5, 1.]]), covSingle = tensor([ [1., .2], [.2, 1.]]), **kwargs):
    residualCovariance = covSingle

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

    ####################### Calculate P(DBoth) given genetic correlation ##############################
    n = N(0, 1)
    thresh1 = n.icdf(pDs[0])
    thresh2 = n.icdf(pDs[1])

    # TODO: I think we need to assert 1 on the diagonal for residualCovariance and covShared

    shared_cov_scaled = covShared * meanEffectCovarianceScale
    residual_cov_scaled = residualCovariance * meanEffectCovarianceScale
    print("PD1 threshold, PD2 threshold", thresh1, thresh2)
    # Interesting; this PDBoth will shrink if there is more correlation between these traits
    # if correlation is 0, then the cdf appears nearly additive, and if correlation close to 1, 
    # the cdf appears nearly that of the larger of the two thresholds
    
    # TODO: I think this must be covShared, where covShared is genetic correlation + environmental
    # otherwise I can get cases where P(V|DBoth,geneBoth) is much smaller than P(V|D1, geneBoth) and P(V|D2, geneBoth), given the exact
    # same covariance
    # TODO: should this be done weighing the different components?
    print(covSingle)
    pdBothGenerator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), residualCovariance))
    PDBoth = tensor(pdBothGenerator.cdf(tensor([thresh1, thresh2])))
    pDsWithBoth = tensor([*pDs, PDBoth])

    print("pDsWithBoth", pDsWithBoth)
    ##################### Calculate effects in genes that affect both conditions #########################
    # No matter how I scale the covariance matrix, correlation will remain the same, great!
    meanEffectsAcrossAllGenes = getTargetMeanEffect(pDs, rrMeans)
    print("meanEffectsAcrossAllGenes", meanEffectsAcrossAllGenes)

    # Covariances scaled to prevent very large mean effects
    effectGenerator = MultivariateNormal(meanEffectsAcrossAllGenes, shared_cov_scaled)
    #dims 20_000 x 2
    allEffects = -effectGenerator.sample([nGenes])
    print("allEffects", allEffects)

    pd1Gen = N(allEffects[:, 0], 1)
    pd2Gen = N(allEffects[:, 1], 1)
    PD1GivenV = pd1Gen.cdf(thresh1) 
    PD2GivenV = pd2Gen.cdf(thresh2)

    print("allEffects[i]", allEffects[0])

    PDBothGivenV = []
    for i in range(nGenes):
        # There may be a vectorized way, but would need to bring scipy's cdf method into pytorch
        # scipy requires ndim == 1 on means
        mvn = MultivariateNormal(allEffects[i], covShared)
        mvnw = WrappedMVN(mvn)

        PDBothGivenV.append(mvnw.cdf(tensor([thresh1, thresh2])))
    PDBothGivenV = tensor(PDBothGivenV)
    print("PDBothGivenV.mean", PDBothGivenV.mean())
    print("PDBothGivenV / PDBoth", (PDBothGivenV / PDBoth).mean())

    pdvsInBoth = torch.stack([PD1GivenV, PD2GivenV, PDBothGivenV]).T

    print("pdsCovarOnMean.mean(0)", pdvsInBoth.mean(0))
    # This has ~0 covariance for singel effets, and ~.6 correlation for one of the single effects with a joint effect
    print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,1])\n", np.corrcoef(pdvsInBoth[:,0], pdvsInBoth[:,1]))
    print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,2])\n", np.corrcoef(pdvsInBoth[:,0], pdvsInBoth[:,2]))

    ############### Calculate effects in genes that affect a single conditions ##################
    effectGenerator= MultivariateNormal(meanEffectsAcrossAllGenes, residual_cov_scaled)
    allEffectsFor12 = -effectGenerator.sample([nGenes])
    pd1Gen = N(allEffectsFor12[:, 0], 1)
    pd2Gen = N(allEffectsFor12[:, 1], 1)
    PD1Vsingle = pd1Gen.cdf(thresh1)
    PD2Vsingle = pd2Gen.cdf(thresh2)

    PDBoth1GivenV = []
    PDBoth2GivenV = []
    for i in range(nGenes):
        mvn = MultivariateNormal(tensor([allEffectsFor12[i, 0], 0]), residualCovariance)
        mvn2 = MultivariateNormal(tensor([0, allEffectsFor12[i, 1]]), residualCovariance)
        mvnw1 = WrappedMVN(mvn)
        mvnw2 = WrappedMVN(mvn2)

        PDBoth1GivenV.append(mvnw1.cdf(tensor([thresh1, thresh2])))
        PDBoth2GivenV.append(mvnw2.cdf(tensor([thresh1, thresh2])))
    PDBoth1GivenV = tensor(PDBoth1GivenV)
    PDBoth2GivenV = tensor(PDBoth2GivenV)

    print("PDBoth1GivenV", PDBoth1GivenV)
    print("PDBoth2GivenV", PDBoth2GivenV)

    print("np.corrcoef(PD1Vsingle, PD2Vsingle)\n", np.corrcoef(PD1Vsingle, PD2Vsingle))
    print("np.corrcoef(PD1Vsingle, PDBoth1GivenV)\n", np.corrcoef(PD1Vsingle, PDBoth1GivenV))
    print("np.corrcoef(PD2Vsingle, PDBoth1GivenV)\n", np.corrcoef(PD2Vsingle, PDBoth1GivenV))
    print("np.corrcoef(PD2Vsingle, PDBoth2GivenV)\n", np.corrcoef(PD2Vsingle, PDBoth2GivenV))

    pdvsGeneAffects1 = torch.stack([PD1Vsingle, pDs[1].expand([nGenes]), PDBoth1GivenV])
    pdvsGeneAffects2 = torch.stack([pDs[0].expand([nGenes]), PD2Vsingle, PDBoth2GivenV])
    pdvsNull = pDsWithBoth.expand(pdvsGeneAffects1.T.shape).T

    print("pdvsGeneAffects1.mean", pdvsGeneAffects1.mean(0))
    afDist = Gamma(concentration=afShape, rate=afShape/afMean)
    afs = afDist.sample([nGenes, ])
    print("afs.dist", afs.mean(), "+/-", afs.std())
    print("afs.shape", afs.shape)
    ############# Our multinomial probabilities are, in the margin P(V|gene) ###############################
    # This is decomposed into P(V|Disesase1)P(Disease1) + P(V|Disease2)P(Disease2) ... for every gene
    # To get P(V|Disease) from P(Disease|V), we note
    # P(D|V)P(V) = P(V|D)P(D), SO P(V|D) = P(D|V)*P(V) / P(D)
    # For every gene we have an allele frequency, P(V), sampled from the gamma distribution
    # And we calculate penetrance ( P(D|V) ) above using the mean effects for each genetic architecture
    # So now we need to multiple by P(V), and divide the result by P(D)
    # This gives our true population estimate
    #########################################################################################################
    pvd_base = torch.stack([pdvsNull, pdvsGeneAffects1, pdvsGeneAffects2, pdvsInBoth.T]).transpose(2, 0).transpose(1,2) / pDsWithBoth
    pvds = afs.unsqueeze(-1).unsqueeze(-1).expand(pvd_base.shape) * pvd_base
    
    print("afs", afs)

    pis = tensor([1 - diseaseFractions.sum(), *diseaseFractions])
    categorySampler = Categorical(pis)
    categories  = categorySampler.sample([nGenes,])

    affectedGenes = []
    unaffectedGenes = []
    altCounts = []
    probs = []
    PVDs = []
    for geneIdx in range(nGenes):
        affects = categories[geneIdx]

        if affects == 0:
            unaffectedGenes.append(geneIdx)
        else:
            while affects - 1 >= len(affectedGenes):
                affectedGenes.append([])
            affectedGenes[affects - 1].append(geneIdx)
        altCountsGene, p, pvnd, pvd = genAlleleCountFromPVDS(nCases = nCases, nCtrls = nCtrls, PVDs = pvds[geneIdx, affects], afMean = afs[geneIdx], pDs = pDsWithBoth)

        altCounts.append(altCountsGene.numpy())
        probs.append(p.numpy())
        PVDs.append([pvnd, *pvd])

    altCounts = tensor(altCounts)
    probs = tensor(probs)
    PVDs = tensor(PVDs)

    return {"altCounts": altCounts, "afs": probs, "categories": categories, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "PDs": pDsWithBoth, "PVDs": PVDs}