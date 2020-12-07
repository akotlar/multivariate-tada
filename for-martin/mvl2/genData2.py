import torch
from torch import Tensor
import torch.tensor as tensor
from torch.distributions import Gamma, Categorical, Categorical, MultivariateNormal, Multinomial, Normal
import numpy as np
from pyper import *
from torch.distributions import MultivariateNormal
from torch import tensor, Tensor
from scipy.stats import multivariate_normal as scimvn
import numpy as np
import torch
from typing import Dict, Tuple, Any
    
class WrappedMVN():
    def __init__(self, mvn: MultivariateNormal):
        self.mvn = mvn
        self.scimvn = scimvn(mean=self.mvn.mean, cov=self.mvn.covariance_matrix)

    def cdf(self, lower: Tensor):
        l = lower.expand(self.mvn.mean.shape)
        return self.scimvn.cdf(l)

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

# TODO: Again check, still not seeing entirely whether I should be sampling mean effects from covariance or correlation matrices
# TODO: values look very fucking weird without r_g ~= 0. Will get P(V|D) of < P(V) for individuals affected by both conditions, for genes that affect only 1 condition
# TOOD: fucking means sometimes shift into the protective realm
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
    fudge_factor: int = 1.,
    **kwargs):
    assert PD.shape[0] == 2

    def get_prevalence(PD: Tensor, r_p: Tensor) -> Tuple[Any, Any]:
        norm = Normal(tensor([0., 0]), 1)
        disease_z_scores = norm.icdf(PD)

        pd_both_generator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), r_p))
        PD_both = tensor(pd_both_generator.cdf(disease_z_scores))
        PD_with_both = tensor([*PD, PD_both])

        return disease_z_scores, PD_with_both

    def get_mean_effects(PD: Tensor, RR: Tensor):
        norm = Normal(tensor([0., 0]), 1)
        PD_z_score = norm.icdf(tensor(1 - PD))
        PDV_z_score = norm.icdf(1-PD * RR)
        mean_effect = PDV_z_score - PD_z_score
        return mean_effect

    # rg = covg/torch.sqrt(hx * hy)
    ####################### Calculate P(DBoth) given genetic correlation ##############################
    disease_z_scores, PD_with_both = get_prevalence(PD, r_p)
    mean_effects = get_mean_effects(PD, RR_mean)

    print("disease_z_scores", disease_z_scores)
    print("PD_with_both", PD_with_both)
    print("mean_effects", mean_effects)

    effect_generator_affects_one = MultivariateNormal(mean_effects, cov_e * fudge_factor)
    effects_generator_affects_both = MultivariateNormal(mean_effects, cov_g * fudge_factor)

    def calc_pdv(mean_effects: Tensor) -> Tensor:
        n = Normal(mean_effects, 1)
        PDV_single = n.cdf(disease_z_scores)
        mvn = WrappedMVN(MultivariateNormal(mean_effects, torch.eye(2)))
        PD12V = mvn.cdf(tensor(disease_z_scores))
        pd = tensor([*PDV_single, PD12V])

        return pd

    pis = tensor([1 - pi.sum(), *pi])
    category_sampler = Categorical(pis)
    categories  = category_sampler.sample([n_genes,])

    PV = Gamma(PV_shape, PV_shape/PV_mean).sample([n_genes,])
    print("PV", PV.mean(0), PV.std(0))
    N = n_cases.sum() + n_ctrls
    PD_hat = n_cases / N
    PND = 1.0 - PD_with_both.sum()
    PNDhat = 1.0 - PD_hat.sum()

    affected_genes = [[], [], []]
    unaffected_genes = []
    alt_counts = []
    PVD_PD_hats = []
    PVDs = []
    def run(geneIdx):
        PDV = None
        PV_gene = PV[geneIdx]
        
        mean_effects = None
        affects = categories[geneIdx] - 1
        if affects == -1:
            unaffected_genes.append(geneIdx)
            PDV = PD_with_both
        else:
            affected_genes[affects - 1].append(geneIdx)
            
            if affects < 2:
                mean_effects = tensor([0., 0.])
                effects = effect_generator_affects_one.sample()
                mean_effects[affects] = effects[affects]
                # print("affects", affects, "effects", effects, "mean_effects", mean_effects)
            else:
                assert affects == 2
                mean_effects = effects_generator_affects_both.sample()

            PDV = calc_pdv(mean_effects)
        
        # P(D|V) = P(V|D)P(D) / P(V)
        PVD_PD = PDV * PV_gene
        PVND_PND_POP = PV_gene - PVD_PD.sum()

        if PVND_PND_POP < 0:
            print("PND", PND)
            print("PV_gene", PV)
            print("PVD_PD_pop_estimate", PVD_PD, "PVD_PD.sum()", PVD_PD.sum())
            print("PVND_PND_POP", PVND_PND_POP)

        assert PVND_PND_POP > 0

        PVD = PVD_PD / PD_with_both
        PVND = PVND_PND_POP / PND

        # print("affects", affects, "PVD", PVD)
        # print("PDV", PDV, PD_with_both)
        # print("affects", affects, "PVD", PVD)

        marginal_count = int(torch.ceil(PVND * n_ctrls + (PVD * n_cases).sum()))
        PVD_PD_hat = tensor([PVND, *PVD]) * tensor([PNDhat, *PD_hat])
        alt_count_gene = Multinomial(probs=PVD_PD_hat, total_count=marginal_count).sample()

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
        "alt_counts": alt_counts, "PV": PV, "PVDs": PVDs, "PD_with_both": PD_with_both, ""
        "PVD_PD_hats": PVD_PD_hats, "categories": categories, "affected_genes": affected_genes,
        "unaffected_genes": unaffected_genes}


# def genParams(pis=tensor([.1, .1, .05]), rrShape=tensor(10.), rrMeans=tensor([3., 3., 1.5]), afShape=tensor(10.), afMean=tensor(1e-4), nCases=tensor([5e3, 5e3, 2e3]), nCtrls=tensor(5e5), covShared=tensor([[1, .5], [.5, 1]]), covSingle=tensor([[1., 0.], [0., 1.]]), meanEffectCovarianceScale=tensor(.01), pDs=None, rrtype="default", **kwargs):
#     nGenes = 20_000

#     assert pDs is not None

#     return [{
#         "nGenes": nGenes,
#         "nCases": nCases,
#         "nCtrls": nCtrls,
#         "pDs": pDs,
#         "diseaseFractions": pis,
#         "rrShape": rrShape,
#         "rrMeans": rrMeans,
#         "afShape": afShape,
#         "afMean": afMean,
#         "covShared": covShared,
#         "covSingle": covSingle,
#         "meanEffectCovarianceScale": meanEffectCovarianceScale,
#         "rrtype": rrtype
#     }]


# def v6liability2(nCases, nCtrls, pDs = tensor([.01, .01]), diseaseFractions = tensor([.05, .05, .01]), rrMeans = tensor([3, 5]), afMean = tensor(1e-4), afShape = tensor(50.), nGenes=20000,
#              meanEffectCovarianceScale=tensor(.01), covShared=tensor([ [1., .5], [.5, 1.]]), covSingle = tensor([ [1., .2], [.2, 1.]]), **kwargs):
#     residualCovariance = covSingle

#     def get_target_mean_effects(PD: Tensor, rrTarget: Tensor):
#         norm = N(0, 1)
#         pdThresh = norm.icdf(1 - PD)
#         pdTarget = PD * rrTarget
#         print("pdThresh", pdThresh)
#         print("pdTarget", pdTarget)
#         pdvthresh = norm.icdf(1 - pdTarget)
#         print("pdvthresh", pdvthresh)
#         meanEffect = pdThresh - pdvthresh
#         print("meanEffect", meanEffect)
#         return meanEffect

#     ####################### Calculate P(DBoth) given genetic correlation ##############################
#     n = N(0, 1)
#     thresh1 = n.icdf(pDs[0])
#     thresh2 = n.icdf(pDs[1])

#     # TODO: I think we need to assert 1 on the diagonal for residualCovariance and covShared

#     shared_cov_scaled = covShared * meanEffectCovarianceScale
#     residual_cov_scaled = residualCovariance * meanEffectCovarianceScale
#     print("PD1 threshold, PD2 threshold", thresh1, thresh2)
#     # Interesting; this PDBoth will shrink if there is more correlation between these traits
#     # if correlation is 0, then the cdf appears nearly additive, and if correlation close to 1, 
#     # the cdf appears nearly that of the larger of the two thresholds
    
#     # TODO: I think this must be covShared, where covShared is genetic correlation + environmental
#     # otherwise I can get cases where P(V|DBoth,geneBoth) is much smaller than P(V|D1, geneBoth) and P(V|D2, geneBoth), given the exact
#     # same covariance
#     # TODO: should this be done weighing the different components?
#     print(covSingle)
#     pdBothGenerator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), residualCovariance))
#     PDBoth = tensor(pdBothGenerator.cdf(tensor([thresh1, thresh2])))
#     pDsWithBoth = tensor([*pDs, PDBoth])

#     print("pDsWithBoth", pDsWithBoth)
#     ##################### Calculate effects in genes that affect both conditions #########################
#     # No matter how I scale the covariance matrix, correlation will remain the same, great!
#     meanEffectsAcrossAllGenes = get_target_mean_effects(pDs, rrMeans)
#     print("meanEffectsAcrossAllGenes", meanEffectsAcrossAllGenes)

#     # Covariances scaled to prevent very large mean effects
#     effectGenerator = MultivariateNormal(meanEffectsAcrossAllGenes, shared_cov_scaled)
#     #dims 20_000 x 2
#     allEffects = -effectGenerator.sample([nGenes])
#     print("allEffects", allEffects)

#     pd1Gen = N(allEffects[:, 0], 1)
#     pd2Gen = N(allEffects[:, 1], 1)
#     PD1GivenV = pd1Gen.cdf(thresh1) 
#     PD2GivenV = pd2Gen.cdf(thresh2)

#     print("allEffects[i]", allEffects[0])

#     PDBothGivenV = []
#     for i in range(nGenes):
#         # There may be a vectorized way, but would need to bring scipy's cdf method into pytorch
#         # scipy requires ndim == 1 on means
#         mvn = MultivariateNormal(allEffects[i], covShared)
#         mvnw = WrappedMVN(mvn)

#         PDBothGivenV.append(mvnw.cdf(tensor([thresh1, thresh2])))
#     PDBothGivenV = tensor(PDBothGivenV)
#     print("PDBothGivenV.mean", PDBothGivenV.mean())
#     print("PDBothGivenV / PDBoth", (PDBothGivenV / PDBoth).mean())

#     pdvsInBoth = torch.stack([PD1GivenV, PD2GivenV, PDBothGivenV]).T

#     print("pdsCovarOnMean.mean(0)", pdvsInBoth.mean(0))
#     # This has ~0 covariance for singel effets, and ~.6 correlation for one of the single effects with a joint effect
#     print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,1])\n", np.corrcoef(pdvsInBoth[:,0], pdvsInBoth[:,1]))
#     print("np.corrcoef(pdvInBoth[:,0], pdvInBoth[:,2])\n", np.corrcoef(pdvsInBoth[:,0], pdvsInBoth[:,2]))

#     ############### Calculate effects in genes that affect a single conditions ##################
#     effectGenerator= MultivariateNormal(meanEffectsAcrossAllGenes, residual_cov_scaled)
#     allEffectsFor12 = -effectGenerator.sample([nGenes])
#     pd1Gen = N(allEffectsFor12[:, 0], 1)
#     pd2Gen = N(allEffectsFor12[:, 1], 1)
#     PD1Vsingle = pd1Gen.cdf(thresh1)
#     PD2Vsingle = pd2Gen.cdf(thresh2)

#     PDBoth1GivenV = []
#     PDBoth2GivenV = []
#     for i in range(nGenes):
#         mvn = MultivariateNormal(tensor([allEffectsFor12[i, 0], 0]), residualCovariance)
#         mvn2 = MultivariateNormal(tensor([0, allEffectsFor12[i, 1]]), residualCovariance)
#         mvnw1 = WrappedMVN(mvn)
#         mvnw2 = WrappedMVN(mvn2)

#         PDBoth1GivenV.append(mvnw1.cdf(tensor([thresh1, thresh2])))
#         PDBoth2GivenV.append(mvnw2.cdf(tensor([thresh1, thresh2])))
#     PDBoth1GivenV = tensor(PDBoth1GivenV)
#     PDBoth2GivenV = tensor(PDBoth2GivenV)

#     print("PDBoth1GivenV", PDBoth1GivenV)
#     print("PDBoth2GivenV", PDBoth2GivenV)

#     print("np.corrcoef(PD1Vsingle, PD2Vsingle)\n", np.corrcoef(PD1Vsingle, PD2Vsingle))
#     print("np.corrcoef(PD1Vsingle, PDBoth1GivenV)\n", np.corrcoef(PD1Vsingle, PDBoth1GivenV))
#     print("np.corrcoef(PD2Vsingle, PDBoth1GivenV)\n", np.corrcoef(PD2Vsingle, PDBoth1GivenV))
#     print("np.corrcoef(PD2Vsingle, PDBoth2GivenV)\n", np.corrcoef(PD2Vsingle, PDBoth2GivenV))

#     pdvsGeneAffects1 = torch.stack([PD1Vsingle, pDs[1].expand([nGenes]), PDBoth1GivenV])
#     pdvsGeneAffects2 = torch.stack([pDs[0].expand([nGenes]), PD2Vsingle, PDBoth2GivenV])
#     pdvsNull = pDsWithBoth.expand(pdvsGeneAffects1.T.shape).T

#     print("pdvsGeneAffects1.mean", pdvsGeneAffects1.mean(0))
#     afDist = Gamma(concentration=afShape, rate=afShape/afMean)
#     afs = afDist.sample([nGenes, ])
#     print("afs.dist", afs.mean(), "+/-", afs.std())
#     print("afs.shape", afs.shape)
#     ############# Our multinomial probabilities are, in the margin P(V|gene) ###############################
#     # This is decomposed into P(V|Disesase1)P(Disease1) + P(V|Disease2)P(Disease2) ... for every gene
#     # To get P(V|Disease) from P(Disease|V), we note
#     # P(D|V)P(V) = P(V|D)P(D), SO P(V|D) = P(D|V)*P(V) / P(D)
#     # For every gene we have an allele frequency, P(V), sampled from the gamma distribution
#     # And we calculate penetrance ( P(D|V) ) above using the mean effects for each genetic architecture
#     # So now we need to multiple by P(V), and divide the result by P(D)
#     # This gives our true population estimate
#     #########################################################################################################
#     pvd_base = torch.stack([pdvsNull, pdvsGeneAffects1, pdvsGeneAffects2, pdvsInBoth.T]).transpose(2, 0).transpose(1,2) / pDsWithBoth
#     pvds = afs.unsqueeze(-1).unsqueeze(-1).expand(pvd_base.shape) * pvd_base
    
#     print("afs", afs)

#     pis = tensor([1 - diseaseFractions.sum(), *diseaseFractions])
#     categorySampler = Categorical(pis)
#     categories  = categorySampler.sample([nGenes,])

#     affectedGenes = []
#     unaffectedGenes = []
#     altCounts = []
#     probs = []
#     PVDs = []
#     for geneIdx in range(nGenes):
#         affects = categories[geneIdx]

#         if affects == 0:
#             unaffectedGenes.append(geneIdx)
#         else:
#             while affects - 1 >= len(affectedGenes):
#                 affectedGenes.append([])
#             affectedGenes[affects - 1].append(geneIdx)
#         altCountsGene, p, pvnd, pvd = genAlleleCountFromPVDS(nCases = nCases, nCtrls = nCtrls, PVDs = pvds[geneIdx, affects], afMean = afs[geneIdx], pDs = pDsWithBoth)

#         altCounts.append(altCountsGene.numpy())
#         probs.append(p.numpy())
#         PVDs.append([pvnd, *pvd])

#     altCounts = tensor(altCounts)
#     probs = tensor(probs)
#     PVDs = tensor(PVDs)

#     return {"altCounts": altCounts, "afs": probs, "categories": categories, "affectedGenes": affectedGenes, "unaffectedGenes": unaffectedGenes, "PDs": pDsWithBoth, "PVDs": PVDs}