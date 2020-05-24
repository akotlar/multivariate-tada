import numpy as np
from torch import tensor
from mvl import genData, likelihoods

from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 3], [5, 5, 5], [5, 5, 1.5]]), pis=tensor([ [.1, .1, .1]]),
                     nCases=tensor([15e3, 15e3, 6e3]), nCtrls=tensor(3e5), nIterations=100)

    np.save("simulation-res-low-5-24-20", res)
