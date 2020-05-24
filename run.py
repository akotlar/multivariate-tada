import numpy as np
from torch import tensor
from mvl import genData, likelihoods

from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    genData.runSimMT(fitMethod="nelder-mead", mt=False, nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.05, .05, .05]]),
                     nCases=tensor([15e3, 15e3, 6e3]), nCtrls=tensor(3e5), nIterations=6)
