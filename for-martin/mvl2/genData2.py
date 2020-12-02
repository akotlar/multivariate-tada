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

def liabilitySumStat(nCases, nCtrls, pDs = tensor([.01, .01]), diseaseFractions = tensor([.05, .05, .01]), rrMeans = tensor([3, 5]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
             geneticVariance=tensor(.08), totalVariance=tensor(.1), geneticCorrelation=tensor([ [1., .5], [.5, 1.]]), residualCorrelation = tensor([ [1., .2], [.2, 1.]]), **kwargs):
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

    residualVariance = totalVariance - geneticVariance
    print("totalVariance", totalVariance, "geneticVariance", geneticVariance, "residualVariance", residualVariance)

    genetic_covariance = geneticCorrelation * geneticVariance
    residual_covariance = residualCorrelation * residualVariance

    pdBothGenerator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), residualCorrelation))
    PDBoth = tensor(pdBothGenerator.cdf(tensor([thresh1, thresh2])))
    pDsWithBoth = tensor([*pDs, PDBoth])

    print("pDsWithBoth", pDsWithBoth)
    ##################### Calculate effects in genes that affect both conditions #########################
    meanEffectsAcrossAllGenes = getTargetMeanEffect(pDs, rrMeans)
    print("meanEffectsAcrossAllGenes", meanEffectsAcrossAllGenes)

    effectGenerator = MultivariateNormal(meanEffectsAcrossAllGenes, genetic_covariance)
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
    effectGenerator= MultivariateNormal(meanEffectsAcrossAllGenes, residual_covariance)
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