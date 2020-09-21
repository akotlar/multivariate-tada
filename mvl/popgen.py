from pyro.distributions import Normal
from torch import tensor
norm = Normal(0, 1)
one = tensor(1.)
def liabilityFromRisk(PDV, PD, PV):
    # PD == K == prevalence of the disease
    # PV == p == frequency of the risk allele
    # prevalence	0.025
    # threshold	normsinv(1-prevalence)
    # maf	1.00E-04
    # Penetrance	0.1
    # Mean effect	threshold - normsinv(1 - penetrance)
    # mu2 	((maf) * Mean effect) / (1-maf)
    threshold  = norm.icdf(1-PD)
    print(threshold)
    meanEffect = threshold - norm.icdf(one - PDV)
    p = maf
    q = 1 - PV
    meanEffect2 = PV * meanEffect / q

    return meanEffect, meanEffect2

