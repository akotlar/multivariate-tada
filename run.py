import numpy as np
from torch import tensor
from mvl import genData, likelihoods
from datetime import date
from torch.multiprocessing import set_start_method
import torch
if __name__ == "__main__":
    set_start_method('spawn', force=True)
    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5], [2, 2, 2], [3, 3, 1.5], [3, 3, 3]]), pis=tensor([[.05, .05, .05], [.1, .1, .1]]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), nIterations=100)

    # np.save("simulation-res-low-5-24-20", res)


    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[5, 5, 5]]), pis=tensor([[.05, .05, .05]]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=10, old=False, runName="new")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[5, 5, 5]]), pis=tensor([[.05, .05, .05]]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=10, old=True, runName="old")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[5, 5, 5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=10, old=False, runName="new-pds-specified")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[5, 5, 5]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-5, nIterations=10, old=True, runName="old-pds-specified")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[10, 10, 10]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=False, runName="new-pds-specified-rr10-af1e-4")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[10, 10, 10]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=True, runName="old-pds-specified-rr10-af1e-4")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[10, 10, 10]]), pis=tensor([[.05, .05, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=True, rrtype="unique", generatingFn=genData.v6, runName="old-pds-specified-rr10-af1e-4-unqique")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 7]]), pis=tensor([[.1, .07, .02]]), pDs=tensor([.01, .01, .002]), covSingle=tensor( [ [.1, 0], [0, .1]]),
    #                  covShared=tensor( [ [.1, .05], [.05, .1]]),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=True, rrtype="lognormal-unique", generatingFn=genData.v6normal, runName="lognormal-unique-cov-.1-.05")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 7]]), pis=tensor([[.1, .07, .02]]), pDs=tensor([.01, .01, .002]), covSingle=tensor( [ [.5, 0], [0, .5]]) * torch.log(tensor(1.0001)),
    #                  covShared=tensor( [ [.5, .25], [.25, .5]]) * torch.log(tensor(1.0001)),
    #                  nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=False, rrtype="lognormal-unique", generatingFn=genData.v6normal, runName="lognormal-unique-cov-.25-.5")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 7, 2]]), pis=tensor([[.1, .07, .02]]),
    #                 pDs=tensor([.01, .01, .002]), 
    #                 nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=True, rrtype="unique", generatingFn=genData.v6normal, runName="new-pds-specified-rr10-af1e-4-unqique")

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 7, 2]]), pis=tensor([[.1, .07, .02]]),
    #                 pDs=tensor([.01, .01, .002]), 
    #                 nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=False, rrtype="unique", generatingFn=genData.v6normal, runName="old-pds-specified-rr10-af1e-4-unqique")


    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[3, 7, 2]]), pis=tensor([[.1, .07, .02]]),
    #                 pDs=tensor([.01, .01, .002]), 
    #                 nCases=tensor([10e3, 10e3, 4e3]), nCtrls=tensor(3e5), afMean = 1e-4, nIterations=10, old=False, rrtype="unique-multiplicative", generatingFn=genData.v6normal, runName="old-pds-specified-rr10-af1e-4-unqique")

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

    # res = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 1.5]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5,1_5,1_5_gamma_15kcases")

    # res1 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5,1_5,0_gamma_15kcases", stacked=False)
    # print('res1-non-stacked', res1)
    # res2 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5,1_5,0_gamma_15kcases_stacked", stacked=True)
    # print('res2-stacked', res2)

    # res1 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.5, 1.5, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(10.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr1_5,1_5,0_gamma_15kcases", stacked=False)
    # print('res1-non-stacked', res1)


    # res1 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[2, 2, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr3,3,0_gamma_15kcases_3e5ctrls", stacked=False)
    # print('res1-non-stacked', res1)
    # res2 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[2, 2, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr3,3,0_gamma_15kcases_3e5ctrls_stacked", stacked=True)
    # res2 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[2, 2, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr3,3,0_gamma_15kcases_3e5ctrls_stacked_dirichlet_mean", stacked=True, piPrior=True)

    # res2 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[2, 2, 2]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([1.5e4, 1.5e4, 6e3]), nCtrls=tensor(3e5), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rr3,3,0_gamma_15kcases_3e5ctrls_stacked",stacked=True)

    # Seems to work less well
    # res3stacked = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.2, 1.2, 1.2]]), pis=tensor([[.1, .05, .02]]),
    #                  nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = .085, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_stacked",stacked=True)


    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.2, 1.2, 1.2]]), pis=tensor([[.1, .05, .02]]),
    #                  nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = .085, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls")

    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.2, 1.2, 1.2]]), pis=tensor([[.2, .07, .05]]),
            #          nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = .085, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
            #  covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            #  covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls")

    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.2, 1.2, 1.2]]), pis=tensor([[.2, .07, .05]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = .085, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_pds_01_01_002")

    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.2, 1.2, 1.2]]), pis=tensor([[.1, .07, .01]]), pDs=tensor([.01, .01, .002]),
    #                  nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_pds_01_01_002_afmean_1e-5")

    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[1.2, 1.2, 0]]), pis=tensor([[.1, .05, .02]]),
    #                  nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = .085, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #          covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #          covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_misspecified")

    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[4, 4, 4]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.04, .04, .01]),
    #                 nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #         covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #         covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_DBS_02_af_7_rr_4_pds_04_04_01")

    # res3 = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[4, 4, 4]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.02, .02, .005]),
    #                 nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #         covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #         covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_DBS_02_af_7_rr")

    # res3misspecified = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[4, 4, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.02, .02, .005]),
    #                 nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #         covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #         covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_DBS_02_af_7_rr_0_shared_rr")

    # res3misspecified = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[4, 4, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                 nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #         covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #         covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_pds_01_01_002_af_7_rr_0_shared_rr")

    # res3misspecified = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrs=tensor([[8, 8, 0]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01, .002]),
    #                 nCases=tensor([4e3, 3.5e3, 1e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6,
    #         covShared=tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    #         covSingle=tensor([[1, 0], [0, 1]]), runName="rrDBS_3k_cases_1kboth_50k_ctrls_pds_01_01_002_af_7_rr8_and_0_shared_rr")

    ########## Liability models #########
    # liabiilitytest = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrMeans=tensor([[3, 3]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01]),
    #                 nCases=tensor([1.5e4, 1.5e4, 4e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6liability,
    #         covShared=tensor([[1, .5], [.5, 1]]), meanEffectCovarianceScale=tensor(.01), runName="liability_rr3")

    # liabiilitytest = genData.runSimMT(fitMethod="nelder-mead", nEpochs=1, rrMeans=tensor([[2, 5]]), pis=tensor([[.1, .05, .02]]), pDs=tensor([.01, .01]),
    #                 nCases=tensor([1.5e4, 1.5e4, 4e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), nIterations=10, nEpochsPerIteration=1, generatingFn=genData.v6liability,
    #         covShared=tensor([[1, .5], [.5, 1]]), meanEffectCovarianceScale=tensor(.01), runName="liability_rr2-5")
            
    liabiilitytest = genData.runSimMT(
                        fitMethod="nelder-mead", nEpochs=1, rrMeans=tensor([[1.5, 1.5]]), pis=tensor([[.1, .1, .1]]), pDs=tensor([.05, .05]),
                        nCases=tensor([1.5e4, 1.5e4, 4e3]), nCtrls=tensor(5e4), afMean = 1e-4, rrShape = tensor(50.), afShape= tensor(50.), nIterations=10,
                        nEpochsPerIteration=1, generatingFn=genData.v6liability,
                        covShared=tensor([[1, .5], [.5, 1]]), meanEffectCovarianceScale=tensor(.01), runName="liability_rr3")