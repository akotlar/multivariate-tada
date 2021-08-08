import torch
from torch import tensor, Tensor
from torch.distributions import Gamma, Categorical, Categorical, MultivariateNormal, Multinomial, Normal
import numpy as np
from pyper import *
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal as scimvn
import numpy as np
import torch
from typing import Dict, Tuple, Any, Optional
    
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

# TODO: Should I be sampling mean effects from covariance or correlation matrices?
# TODO: Values look very fucking weird without r_g ~= 0. Will get P(V|D) of < P(V) for individuals affected by both conditions, for genes that affect only 1 condition
# TODO: means sometimes shift into the protective realm
def gen_counts(
    n_cases: Tensor,
    n_ctrls: Tensor,
    pi: Tensor,
    PD: Tensor,
    RR_mean: Tensor,
    PV_mean: Tensor,
    r_p: Tensor,
    cov_g: Tensor,
    cov_e: Tensor,
    n_genes: int = 20000,
    fudge_factor: int = 1.,
    PV_shape: Optional[Tensor] = None,
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

    # print("disease_z_scores", disease_z_scores)
    # print("PD_with_both", PD_with_both)
    # print("mean_effects", mean_effects)

    effect_generator_affects_one = MultivariateNormal(mean_effects, cov_e * fudge_factor)
    effects_generator_affects_both = MultivariateNormal(mean_effects, cov_g * fudge_factor)

    def calc_pdv(mean_effects: Tensor) -> Tensor:
        n = Normal(mean_effects, 1)
        # print("disease_z_scores:", disease_z_scores)
        PDV_single = n.cdf(disease_z_scores)
        mvn = WrappedMVN(MultivariateNormal(mean_effects, torch.eye(2)))
        PD12V = mvn.cdf(tensor(disease_z_scores))
        pd = tensor([*PDV_single, PD12V])
        # print("pd:", pd)

        return pd

    pis = tensor([1 - pi.sum(), *pi])
    category_sampler = Categorical(pis)
    categories  = category_sampler.sample([n_genes,])

    PV = None
    if PV_shape is None:
        PV = tensor(PV_mean).expand([n_genes,])
    else:
        PV = Gamma(PV_shape, PV_shape/PV_mean).sample([n_genes,])

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