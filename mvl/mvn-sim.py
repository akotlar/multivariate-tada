from torch import Tensor, tensor
import torch
from torch.distributions import Uniform, MultivariateNormal, Bernoulli, Categorical
import numpy as np

def genData(kConditions: int = 2, nIndividals: int = 1000, pis = tensor([.05, .05, .01])) -> Tensor:
    assert(kConditions == 2)

    # For our test, 2 diseases, with some small effect size
    # features are: effect on disease 1, cov, cov, effect on disease 2

    # TODO: do we need environmental covariance? I think no; it's random contribution
    # TODO: This yields absolutely huge risks....
    # h0cov = tensor([
    #     [1, 0.],
    #     [0, 1]])
    # h1cov = tensor([
    #     [1.01, 0],
    #     [0, 1]])
    # h2cov = tensor([
    #     [1, 0],
    #     [0, 1.01]])
    # hBothCov = tensor([
    #     [1.01, .4],
    #     [.4, 1.01]])
    h0cov = tensor([
        [1, 0.],
        [0, 1]])
    h1cov = tensor([
        [1.005, 0],
        [0, 1]])
    h2cov = tensor([
        [1, 0],
        [0, 1.005]])
    hBothCov = tensor([
        [1.005, .4],
        [.4, 1.005]])
    covs = torch.stack([h0cov, h1cov, h2cov, hBothCov]).numpy()

    nGenes = 20_000
    nVariantsPerGene = 100
    nVariants = nGenes*nVariantsPerGene*kConditions 

    dist = Uniform(0, .1)
    p = dist.sample([nGenes, nVariantsPerGene, kConditions])
    q = (1-p)

    pisWithNull = tensor([1 - pis.sum(), *pis])
    covIndicesSampler = Categorical(pisWithNull)
    covIndices = covIndicesSampler.sample([nGenes,])
    covMatrix = []
    for idx in covIndices:
        covMatrix.append(covs[idx])
    covMatrix = tensor(covMatrix)
    covMatrix = covMatrix.unsqueeze(1).expand([nGenes, nVariantsPerGene, kConditions, kConditions])
    means = tensor([0.]).expand([nGenes, nVariantsPerGene, kConditions])
    # Initial version used a random covariance matrix, this worked great.
    # covMatrix = torch.rand(nGenes, nVariantsPerGene, kConditions, kConditions)
    # covMatrix = torch.matmul(covMatrix, covMatrix.transpose(2,3)).add_(torch.eye(kConditions))

    # Where we have the issue
    effects = MultivariateNormal(means, covMatrix)  

    geneProportions = (nGenes * pis).type(torch.int32)

    def effects2():
        allele_1_eff = effects.sample()
        p_eff = allele_1_eff*p
        allele_2_eff = - p_eff / q

        return allele_1_eff, allele_2_eff

    ef = effects2()
    eff3 = torch.stack([ef[0], ef[1]], 3)
    eff4 = eff3.reshape([nVariants, 2])
    np.array_equal(eff3.reshape([nVariants, 2]).reshape([nGenes, nVariantsPerGene, kConditions, 2]), eff3)   

    # which gene do we sample
    categorySampler = Bernoulli(p)

    individuals = []

    for ind in range(nIndividals):
        a1Indices = categorySampler.sample().type(torch.int64).flatten().view(-1, 1)
        a2ndices = categorySampler.sample().type(torch.int64).flatten().view(-1, 1)

        allele1 = eff4.gather(1, a1Indices).reshape([nGenes, nVariantsPerGene, kConditions])
        allele2 = eff4.gather(1, a2ndices).reshape([nGenes, nVariantsPerGene, kConditions])
        personGenotypeEffects = torch.stack([allele1, allele2], dim=-1)
        
        # mean phenotype j, sum of effects across all alleles on all chromosomes
        # this is dimension k
        mu_j = personGenotypeEffects.sum([0,1,3])
        print(f"ind {ind}, mu_j: {mu_j}")
        individuals.append(personGenotypeEffects)

    return individuals