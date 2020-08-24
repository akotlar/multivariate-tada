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


    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[20, 15, 5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 2e4]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=100)

    
    # # Test effect of covariance
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 3]]), pis=tensor([[.01, .01, .01]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, 
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="covariance0")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 3]]), pis=tensor([[.01, .01, .01]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, 
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="covariance_50%_inBoth")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=100, 
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr3_pi_05")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[6, 6, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=100, nEpochsPerIteration=1,
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr6_pi_05_wide_range_alphas_initialization")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 2e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=100, nEpochsPerIteration=1,
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_up_to_1e6_range_alphas_initialization")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1,
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_2")

    # performs much better than v6normal, but only really when the 0 component is shared...maybe variance is too high in  the normal setting?
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6gen")
    
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[1, .5, .5], [.5, 1, .5], [.5, .5, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6normal_cov_.5")

    # if using 1, 1, 1, rr generation fails: "Generating error: Could not infer dtype of NoneType"
    # Interestingly using relative risks that covary more closely doesn't give any benefits (or not large ones) over 0 covariance
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[1, .9, .9], [.9, 1, .9], [.9, .9, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6normal_cov_.9")

    # let's try a smaller variance, same mean
    # yep, this improves the results
    # though odd, this results in corrcoef  1 1 for 1 & 2 and 2 & 3 in shared risk
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[.2, .9, .9], [.9, .2, .9], [.9, .9, .2]]),
    #          covSingle=tensor([[.2, 0], [0, .2]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6normal_cov_.9_var_.2")

    # # let's try a smaller variance, same mean
    # # what about .2 covariance in all components of shared
    # # ah this also provides 1 corr  coef
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[.2, .2, .2], [.2, .2, .2], [.2, .2, .2]]),
    #          covSingle=tensor([[.2, 0], [0, .2]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6normal_cov_.2_var_.2")

    # let's try with covariacne < variance (say half, so that 2*Cov == variance)
    # this also works much better than 1 var 0 cov
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[.2, .1, .1], [.1, .2, .1], [.1, .1, .2]]),
    #          covSingle=tensor([[.2, 0], [0, .2]]), runName="0_mean_in_shared_component_rr3_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6normal_cov_.1_var_.2")
    
    # The total rr with 3, 3, 3 was 9 for the casesBoth, lets scale the other two compnents to match
    # this does indeed provide excellent estiamtes
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[4.5, 4.5, 0]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([5e4, 5e4, 5e4]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="0_mean_in_shared_component_rr4.5_pi_05_n5e4_and_n5e4_both_normal_range_alphas_initialization_v6normal")

    # The total rr with 3, 3, 3 was 9 for the casesBoth, lets scale the other two compnents to match
    # this does indeed provide excellent estiamtes
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6normal,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr15")

#     res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
#                      nCases=tensor([1e4, 1e4, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
#              covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
#              covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5_gamma")
    
#     res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
#                      nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
#              covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
#              covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5_gamma_15kcases")

#     res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 3, 1.5]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
#                      nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
#              covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
#              covSingle=tensor([[1, 0], [0, 1]]), runName="rr3,3,1_5_gamma_15kcases")

    res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
                     nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
             covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
             covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5,1_5,1_5_gamma_15kcases")