import sys
from torch import tensor, Tensor
from torch.distributions import MultivariateNormal
from scipy.stats import multivariate_normal as scimvn

class WrappedMVN():
    def __init__(self, mvn: MultivariateNormal):
        self.mvn = mvn
        self.scimvn = scimvn(mean=self.mvn.mean, cov=self.mvn.covariance_matrix)

    def cdf(self, lower: Tensor):
        l = lower.expand(self.mvn.mean.shape)
        return self.scimvn.cdf(l)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("thresh1 thresh2 phenotypic_correlation")
        sys.exit(0)

    z_score_1 = -float(sys.argv[1])
    z_score_2 = -float(sys.argv[2])
    rho_p = float(sys.argv[3])

    r_p = tensor([[1., rho_p], [rho_p, 1.]])

    pd_both_generator = WrappedMVN(MultivariateNormal(tensor([0., 0.]), r_p))
    PD_both = pd_both_generator.cdf(tensor([z_score_1, z_score_2]))

    print(PD_both)