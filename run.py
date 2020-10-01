import numpy as np
from torch import tensor
from mvl import genData, likelihoods
from datetime import date
from torch.multiprocessing import set_start_method
import torch
if __name__ == "__main__":
    set_start_method('spawn', force=True)

    liabiilitytest = genData.runSimMT(
                        fitMethod="nelder-mead", nEpochs=1, rrMeans=tensor([[1.5, 1.5]]), pis=tensor([[.1, .1, .1]]), pDs=tensor([.05, .05]),
                        nCases=tensor([1.5e4, 1.5e4, 4e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), afShape= tensor(50.), nIterations=10,
                        nEpochsPerIteration=1, generatingFn=genData.v6liability,
                        covShared=tensor([[1, .5], [.5, 1]]), meanEffectCovarianceScale=tensor(.01), runName="liability_rr1_5,1_5")