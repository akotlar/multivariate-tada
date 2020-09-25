# Uncomment to install necessary libraries
# !pip install torch
# !pip install numpy

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch numpy"])

from torch.distributions import MultivariateNormal as MVN
from torch.distributions import Normal as N
from torch import Tensor, tensor
import numpy as np

def sampleNonNegative(mean: Tensor, cov: Tensor):
    mvn = MVN(mean, cov)
    samples = mvn.sample()

    for sample in samples:
        if sample < 0:
            return sampleNonNegative(mean, cov)

    return samples

def sampleNnonNeg(N: int, mean: Tensor, cov: Tensor):
    print("mean", mean)
    res = []
    for i in range(N):
        n = sampleNonNegative(mean, cov)
        res.append(n.numpy())

    return tensor(res)


def getTargetMeanEffect(PD: Tensor, rrTarget: Tensor):
    norm = N(0, 1)
    pdThresh = norm.icdf(1 - PD)
    pdTarget = PD * rrTarget
    pdvthresh = norm.icdf(1 - pdTarget)
    meanEffect = pdThresh - pdvthresh

    print("pdThresh", pdThresh)
    print("pdTarget", pdTarget)
    print("pdvthresh", pdvthresh)
    print("meanEffect", meanEffect)

    return meanEffect

corrMatrx = tensor( [ [1, .4], [.4, 1] ])
covMatrix = corrMatrx * .1

norm = N(0, 1)
pdBoth = tensor(0.002)
pds = tensor([.01, .01])
rrMeans = tensor([3, 5])

targetMeanEffects = getTargetMeanEffect(pds, rrMeans)

testSampleRestricted = sampleNnonNeg(1000, targetMeanEffects, covMatrix)
totalRestricted = testSampleRestricted[:, 0] + testSampleRestricted[:, 1]

pdsThresh = norm.icdf(1 - pds)
pdBothThresh = norm.icdf(1 - pdBoth)
PDonlyOneGivenVthreshold = pdsThresh - testSampleRestricted
PDBothGivenVthreshold = pdBothThresh - totalRestricted

PDoneGivenV = 1 - norm.cdf(PDonlyOneGivenVthreshold)
PDbothGivenV = 1 - norm.cdf(PDBothGivenVthreshold)

print("pdsThresh", pdsThresh)
print("testSampleRestricted", testSampleRestricted)
print("PDonlyOneGivenVthreshold", PDonlyOneGivenVthreshold)
print("PDoneGivenV.mean(0) / pds", PDoneGivenV.mean(0) / pds)
print("PDbothGivenV.mean(0) / pdBoth", PDbothGivenV.mean(0) / pdBoth)