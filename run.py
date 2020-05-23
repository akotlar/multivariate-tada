import numpy as np

from mvl import genData, likelihoods

from torch.multiprocessing import set_start_method

if  __name__ == "__main__":
    set_start_method('spawn', force=True)
    genData.runSim(fitMethod="nelder-mead", mt=False, nEpochs = 12)