from .likelihoods import nullLikelihood, effect1Likelihood, effect2Likelihood, effectBothLikelihood

# def bayesFactor(altCounts, sampleSizes, )


def bayesFactor(n, altCount, pDs, pis, alphas):
#     print("params", n, altCount, pDs, pis, alphas)
    lNull = nullLikelihood(pDs, altCount)
    l1 = effect1Likelihood(n, pDs, alphas[0], alphas[1], altCount)
    l2 = effect2Likelihood(n, pDs, alphas[0], alphas[2], altCount)
    lBoth = effectBothLikelihood(n, pDs, alphas[0], alphas[1], alphas[2], alphas[3], altCount)

    # these all have the same denominator, seems like I shoudl be able to add them?
    # TODO: why does tada use the  formula:
    #   marglik0.cc <- evidence.null.cc(x.cc, n.cc, rho0, nu0)
    #   marglik1.cc <- evidence.alt.cc(x.cc, n.cc, gamma.cc, beta.cc, rho1, nu1)
    #   BF.cn <- marglik1.cc$cn / marglik0.cc$cn
    #   BF.ca <- marglik1.cc$ca / marglik0.cc$ca
    #   BF <- BF.cn * BF.ca
    bf1 = (l1/lNull).numpy()
    bf2 = (l2/lNull).numpy()
    bf3 = (lBoth/lNull).numpy()
    if bf1 > bf2 and bf1 > bf3:
        maxBF = "bf1"
    elif bf2 > bf1 and bf2 > bf3:
        maxBF = "bf2"
    else:
        maxBF = "bfBoth"
    print("bfs are: ", bf1, bf2, bf3, "max is", maxBF)
    bf  =  bf1 * bf2 * bf3
#     print("BF is", bf)
    return bf
