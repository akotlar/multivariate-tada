import numpy as np
from torch import tensor
from mvl import genData, likelihoods
from datetime import date
from torch.multiprocessing import set_start_method

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5], [2, 2, 2], [3, 3, 1.5], [3, 3, 3]]), pis=tensor([[.05, .05, .05], [.1, .1, .1]]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), nIterations=100)

    # np.save("simulation-res-low-5-24-20", res)


    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[10, 10, 10], [20, 20, 20]]), pis=tensor([[.005, .005, .005], [.01, .01, .01]]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=100)

    # np.save("simulation-res-rr10-5-24-20", res)

    # this is noisy; P(D|V)'s inferred ok, but pi's get extremes
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[20, 15, 5]]), pis=tensor([[.01, .01, .01]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 2e4]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=100)


    res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[20, 15, 5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
                     nCases=tensor([5e4, 5e4, 2e4]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=100)
    np.save(f"test-blah-res-{date.today().__str__()}", res)
