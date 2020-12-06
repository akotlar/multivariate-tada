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
from typing import Dict
    
class WrappedMVN():
    def __init__(self, mvn: MultivariateNormal):
        self.mvn = mvn
        self.scimvn = scimvn(mean=self.mvn.mean, cov=self.mvn.covariance_matrix)

    def cdf(self, lower: Tensor):
        l = lower.expand(self.mvn.mean.shape)
        return self.scimvn.cdf(l)

def gene_alt_counts_from_PVD(n_cases: Tensor, n_ctrls: int, PVD: Tensor, PV_gene: int, PD: Tensor):
    """
    Starting from the true population estimate, P(V|D) we generate the in-sample P(D|V), and use that as our multinomial allele frequecny
    This value is approximately rr*P(D)
    We cannot simply multiply P(V|D) * P(D_hat) because the result may be larger than P(V)
    Instead we need to normalize by the difference between P(D_hat) and P(D)
    P(V|D) * P(D_hat) * P(D) / P(D_hat)? No, P(D|V) is exclusive of P(D)
    It is only later, in inference that we need to re-scale

    Generates 1 pooled control population
    """
    assert n_cases.shape == PVD.shape and n_cases.shape == PD.shape

    N = n_cases.sum() + n_ctrls
    PD_hat = n_cases / N

    PND = 1.0 - PD.sum()
    PNDhat = 1.0 - PD_hat.sum()
    PVD_PD = (PVD * PD)

    PVND_PND_POP = PV_gene - PVD_PD.sum()

    if PVND_PND_POP < 0:
        print("PND", PND)
        print("PV_gene", PV_gene)
        print("PVD", PVD)
        print("PVD_PD_pop_estimate", PVD_PD, "PVD_PD.sum()", PVD_PD.sum())
        print("PVND_PND_POP", PVND_PND_POP)

    assert PVND_PND_POP > 0

    PVND = PVND_PND_POP / PND

    marginalAltCount = int(torch.ceil(PVND * n_ctrls + (PVD * n_cases).sum()))
    PVD_PD_hat = tensor([PVND, *PVD]) * tensor([PNDhat, *PD_hat])
    return Multinomial(probs=PVD_PD_hat, total_count=marginalAltCount).sample(), PVD_PD_hat, PVND, PVD

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

################# TODO: support > 2 conditions ##################
# but from https://www.jstor.org/stable/pdf/2527838.pdf
# r_p = r_g*sqrt(h1^2 * h2^2) + r_e*sqrt((1-h1^2)*(1-h2^2))
# so r_e = (r_p - r_g*sqrt(h1 * h2))/sqrt((1-h1)*(1-h2))
def get_popgen_param(
h2 = tensor([.8, .8]),
v_p = tensor([.01, .01]),
r_p = tensor([[1., 5.], [.5, 1.]]),
r_g = tensor([[1., 5.], [.5, 1.]])) -> Dict[str, torch.Tensor]:
    n = r_g.shape[0]
    assert n == r_g.shape[1] and n == r_p.shape[0] and n == r_p.shape[1]
    assert n == 2

    r_g_vec = r_g.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)
    r_p_vec = r_p.masked_select(~torch.eye(n, dtype=bool)).view(n, n - 1)
    assert r_g_vec[0] == r_g_vec[1]
    assert r_p_vec[0] == r_p_vec[1]

    r_e_vec = (r_p_vec[0] - (r_g_vec[0] * torch.sqrt(torch.prod(h2))) * torch.sqrt(torch.prod(1-h2))).expand(2)
    r_e = torch.eye(2)
    r_e[0,1] = r_e_vec[0]
    r_e[1,0] = r_e_vec[1]

    v_g = v_p * h2
    v_e = v_p - v_g
    pd = torch.diag(v_p ** .5)
    cov_p = pd @ r_p @ pd

    gd = torch.diag(v_g) ** .5
    cov_g = gd @ r_g @ gd

    ed = torch.diag(v_e ** .5) 
    cov_e = ed @ r_e @ ed

    return {
        "r_p": r_p, "r_g": r_g, "r_e": r_e,
        "v_p": v_p, "v_g": v_g, "v_e": v_e,
        "cov_p": cov_p, "cov_g": cov_g, "cov_e": cov_e
    }

def gen_counts(
    n_cases: Tensor,
    n_ctrls: Tensor,
    pi: Tensor,
    PD: Tensor,
    RR_mean: Tensor,
    PV_mean: Tensor,
    PV_shape: Tensor,
    r_p: Tensor,
    r_g: Tensor,
    r_e: Tensor,
    cov_p: Tensor,
    cov_g: Tensor,
    cov_e: Tensor,
    n_genes: int = 20000,
    **kwargs):
    assert PD.shape[0] == 2

    def get_target_mean_effects(PD: Tensor, rr_target: Tensor):
        norm1 = N(0, cov_p[0,0])
        norm2 = N(0, cov_p[1,1])
        pd_threshold1 = norm1.icdf(1 - PD[0])
        pd_threshold2 = norm2.icdf(1 - PD[1])
        pd_target = PD * rr_target
        print("pd_target", pd_target)
        pdv_threshold1 = norm1.icdf(1 - pd_target[0])
        pdv_threshold2 = norm2.icdf(1 - pd_target[1])
        mean_effect = tensor([pd_threshold1 - pdv_threshold1, pd_threshold2 - pdv_threshold2])
        return mean_effect
    print("cov_p[0,0]", cov_p[0,0])
    print("cov_p[1,1]", cov_p[1,1])
    # rg = covg/torch.sqrt(hx * hy)
    ####################### Calculate P(DBoth) given genetic correlation ##############################
    n1= N(0, cov_p[0,0])
    thresh1 = n1.icdf(PD[0])
    n2 = N(0, cov_p[1,1])
    thresh2 = n2.icdf(PD[1])

    print("thresholds 1&2", thresh1, thresh2)

    # TODO: should this be phenotypic correlation or residual?
    # I think prevalence should be due to both due to genetic and environmental reasons
    pd_both_generator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), cov_p))
    PD_both = tensor(pd_both_generator.cdf(tensor([thresh1, thresh2])))
    PD_with_both = tensor([*PD, PD_both])

    print("PD_with_both", PD_with_both)
    ##################### Calculate effects in genes that affect both conditions #########################
    mean_effects = get_target_mean_effects(PD, RR_mean)
    print("mean_effects", mean_effects)

    effects_generator_affects_both = MultivariateNormal(mean_effects, cov_g)
    # Shape nGenes x nIndependentEffects
    effects_given_affects_both = -effects_generator_affects_both.sample([n_genes])

    # TODO: why is this sampling with variance 1?
    PD1V_gen = N(effects_given_affects_both[:, 0], 1)
    PD2V_gen = N(effects_given_affects_both[:, 1], 1)
    PD1V_given_affects_both = PD1V_gen.cdf(thresh1) 
    PD2V_given_affects_both = PD2V_gen.cdf(thresh2)

    PD12V_given_affects_both = []
    for i in range(n_genes):
        mvn = MultivariateNormal(effects_given_affects_both[i], torch.eye(2))
        mvnw = WrappedMVN(mvn)

        PD12V_given_affects_both.append(mvnw.cdf(tensor([thresh1, thresh2])))
    PD12V_given_affects_both = tensor(PD12V_given_affects_both)
    PDV_gene_affects_both = torch.stack([PD1V_given_affects_both, PD2V_given_affects_both, PD12V_given_affects_both]).T

    print("effects_given_affects_both", effects_given_affects_both)
    print("PD12V_given_affects_both.mean", PD12V_given_affects_both.mean())
    print("PD12V_given_affects_both / PDBoth", (PD12V_given_affects_both / PD_both).mean())
    print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,1])\n", np.corrcoef(PDV_gene_affects_both[:,0], PDV_gene_affects_both[:,1]))
    print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,2])\n", np.corrcoef(PDV_gene_affects_both[:,0], PDV_gene_affects_both[:,2]))

    ############### Calculate effects in genes that affect a single conditions ##################
    effect_generator_affects_one = MultivariateNormal(mean_effects, cov_e)
    effects_given_affects_one = -effect_generator_affects_one.sample([n_genes])
    PD1V_gen = N(effects_given_affects_one[:, 0], 1)
    PD2V_gen = N(effects_given_affects_one[:, 1], 1)
    PD1V_given_affects_1 = PD1V_gen.cdf(thresh1)
    PD2V_given_affects_2 = PD2V_gen.cdf(thresh2)

    PD12V_given_affects_1 = []
    PD12V_given_affects_2 = []
    for i in range(n_genes):
        mvn = MultivariateNormal(tensor([effects_given_affects_one[i, 0], 0]), torch.eye(2))
        mvn2 = MultivariateNormal(tensor([0, effects_given_affects_one[i, 1]]), torch.eye(2))
        mvnw1 = WrappedMVN(mvn)
        mvnw2 = WrappedMVN(mvn2)

        PD12V_given_affects_1.append(mvnw1.cdf(tensor([thresh1, thresh2])))
        PD12V_given_affects_2.append(mvnw2.cdf(tensor([thresh1, thresh2])))
    PD12V_given_affects_1 = tensor(PD12V_given_affects_1)
    PD12V_given_affects_2 = tensor(PD12V_given_affects_2)

    print("PD12V_given_affects_1", PD12V_given_affects_1)
    print("PD12V_given_affects_2", PD12V_given_affects_2)

    print("np.corrcoef(PD1V_given_affects_1, PD2V_given_affects_2)\n", np.corrcoef(PD1V_given_affects_1, PD2V_given_affects_2))
    print("np.corrcoef(PD1V_given_affects_1, PD12V_given_affects_1)\n", np.corrcoef(PD1V_given_affects_1, PD12V_given_affects_1))
    print("np.corrcoef(PD2V_given_affects_2, PD12V_given_affects_1)\n", np.corrcoef(PD2V_given_affects_2, PD12V_given_affects_1))
    print("np.corrcoef(PD2V_given_affects_2, PD12V_given_affects_2)\n", np.corrcoef(PD2V_given_affects_2, PD12V_given_affects_2))

    PDV_gene_affects_1 = torch.stack([PD1V_given_affects_1, PD[1].expand([n_genes]), PD12V_given_affects_1])
    PDV_gene_affects_2 = torch.stack([PD[0].expand([n_genes]), PD2V_given_affects_2, PD12V_given_affects_2])
    PDV_gene_affects_none = PD_with_both.expand(PDV_gene_affects_1.T.shape).T

    #PV_dist = Gamma(concentration=PV_shape, rate=PV_shape/PV_mean)
    PV = PV_mean.expand(n_genes)#PV_dist.sample([n_genes, ])
    
    print("PDV_gene_affects_1.mean", PDV_gene_affects_1.mean(0))
    print("PV.dist", PV.mean(), "+/-", PV.std())
    print("PV.shape", PV.shape)
    ############# Our multinomial probabilities are, in the margin P(V|D,gene) ###############################
    # This is decomposed into P(V|Disesase1)P(Disease1) + P(V|Disease2)P(Disease2) ... for every gene
    # To get P(V|Disease) from P(Disease|V), we note
    # P(D|V)P(V) = P(V|D)P(D)
    # P(D|V) = P(V|D)P(D)/P(V)
    # P(D|V)*P(V)/P(D) = P(V|D)
    # For every gene we have an allele frequency, P(V), sampled from the gamma distribution
    # Calculate penetrance P(D|V) using the mean effects for each genetic architecture
    # To get the inverse, P(V|D), use bayes theorem
    #########################################################################################################
    PDV= torch.stack([PDV_gene_affects_none, PDV_gene_affects_1, PDV_gene_affects_2, PDV_gene_affects_both.T]).transpose(2, 0).transpose(1,2)
    print("PDV", PDV)
    PV_expanded = PV.unsqueeze(-1).unsqueeze(-1).expand(PDV.shape)
    print("PV_expanded", PV_expanded)
    PVD_possible = PDV * PV_expanded / PD_with_both
    print("PVD_possible", PVD_possible)
    pis = tensor([1 - pi.sum(), *pi])
    category_sampler = Categorical(pis)
    categories  = category_sampler.sample([n_genes,])

    affected_genes = []
    unaffected_genes = []
    alt_counts = []
    PVD_PD_hats = []
    PVDs = []
    def run(geneIdx):
        affects = categories[geneIdx]

        if affects == 0:
            unaffected_genes.append(geneIdx)
        else:
            while affects - 1 >= len(affected_genes):
                affected_genes.append([])
            affected_genes[affects - 1].append(geneIdx)

        alt_count_gene, PVD_PD_hat, PVND, PVD = gene_alt_counts_from_PVD(n_cases = n_cases, n_ctrls = n_ctrls, PVD = PVD_possible[geneIdx, affects], PV_gene = PV[geneIdx], PD = PD_with_both)

        alt_counts.append(alt_count_gene.numpy())
        PVD_PD_hats.append(PVD_PD_hat.numpy())
        PVDs.append([PVND, *PVD])

    for geneIdx in range(n_genes):
        try:
            run(geneIdx)
        except Exception:
            print(f"failed on {geneIdx}, retrying")
            run(geneIdx)
        

    alt_counts = tensor(alt_counts)
    PVD_PD_hats = tensor(PVD_PD_hats)
    PVDs = tensor(PVDs)

    return {
        "alt_counts": alt_counts, "PVDs": PVDs, "PD_with_both": PD_with_both,
        "PVD_PD_hats": PVD_PD_hats, "categories": categories, "affected_genes": affected_genes,
        "unaffected_genes": unaffected_genes}

def v6liability2(nCases, nCtrls, pDs = tensor([.01, .01]), diseaseFractions = tensor([.05, .05, .01]), rrMeans = tensor([3, 5]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
             meanEffectCovarianceScale=tensor(.01), covShared=tensor([ [1., .5], [.5, 1.]]), covSingle = tensor([ [1., .2], [.2, 1.]]), **kwargs):
    residualCovariance = covSingle

    def get_target_mean_effects(PD: Tensor, rrTarget: Tensor):
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
    meanEffectsAcrossAllGenes = get_target_mean_effects(pDs, rrMeans)
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