probVariantGivenDisease = function(rr, pVariant) {
  (rr * pVariant) / (rr * pVariant + (1 - pVariant))
}

probVariantGivenNotDisease = function(prevalence, pVariant, pVariantGivenDisease) {
  prob <- (pVariant - prevalence*pVariantGivenDisease) / (1 - prevalence)
  if(prob < 0){
    print(paste("pVariant", pVariant, "pVariantGivenDisease", pVariantGivenDisease, "prevalence", prevalence))
    stop("invalid combination of prevalence, allele freq, and case allele freq")
  }else{
    return(prob)
  }
}

likelihoodUnivariateSingleGene = function(xCtrl, xCase1, prevalence1, pi0, pi1, pDiseaseGivenVariant) {
  n = xCtrl + xCase1
  pi0 * dbinom(x = xCase1, size = n, p = prevalence1) + pi1 * dbinom(x = xCase1, size = n, p = pDiseaseGivenVariant) 
}

likelihoodUnivariate = function(xCtrlAllGenes, xCase1AllGenes, prevalence1) {
  function(params) {
    pDiseasesGivenVariant = params[["pDiseaseGivenVariant"]]
    pi1 = params[["pi1"]]
    pi0 = 1 - pi1
    
    if(pDiseasesGivenVariant >= 1 || pDiseasesGivenVariant <= 0 || pi1 <= 0 || pi1 >= 1) {
      return(-Inf)
    }
    
    likelihood = 0
    penaltyCount = length(xCtrlAllGenes)
    for(i in 1:length(xCtrlAllGenes)) {
      ctrlCount = xCtrlAllGenes[i]
      caseCount = xCase1AllGenes[i]
      if(ctrlCount == 0 && caseCount == 0) {
        next
      }
      # likelihoodUnivariateSingleGene = function(xCtrl, xCase1, prevalence1, pi0, pi1, pDiseaseGivenVariant)
      like = likelihoodUnivariateSingleGene(ctrlCount, caseCount, prevalence1, pi0, pi1, pDiseasesGivenVariant)
      
      if(is.nan(like) || like == 0) {
        print(paste("pi1", pi1, "pDiseasesGivenVariant", pDiseasesGivenVariant, "gene", i, "ctrlCount", ctrlCount, "caseCount", caseCount, "res:", likelihood, "log likelihood:",log(like), sep=" "))
        #return(-Inf)
        penaltyCount = penaltyCount - 1
        next
      }
      # print(paste("ctrlCount", ctrlCount, "caseCount", caseCount, "res:", likelihood, "log likelihood:",log(likelihood), sep=" "))
      likelihood = likelihood + log(like)
    }
    
    if(penaltyCount == 0) {
      penaltyCount = 1
    }
    
    likelihood * (length(xCtrlAllGenes) / penaltyCount)
  }
}

# Note:
# genDataFn = function(diseaseFraction, afMean, afShape, rrMean, rrShape, prevalence1 = .01, nCase1 = 1e5, nCtrl1 = 1e5, nGenes = 20000, ctrlRRShape = 1) {
#   rrGenes = vector(length = nGenes)
#   afGenes = vector(length = nGenes)
#   pVariantsGivenDisease1ByGene = vector(length = nGenes)
#   pVariantsGivenNoDiseaseByGene = vector(length = nGenes)
#   case1AlleleCountsByGene = vector(length = nGenes)
#   ctrl1AlleleCountsByGene = vector(length = nGenes)
#   print(paste("PARAMS: ", diseaseFraction, afMean, afShape, rrMean, prevalence1, nCase1, nCtrl1, nGenes, ctrlRRShape))
#   
#   afRate = afShape / afMean
#   rrRate = rrShape / rrMean
#   ctrlRRrate = ctrlRRShape
#   for(i in 1:nGenes){
#     isDisease = 0
#     afGenes[i] = rgamma(1, afShape, rate = afRate)
#     
#     if(i <= (nGenes*diseaseFraction)){
#       isDisease = 1
#       rrGenes[i] = rgamma(1, shape = rrShape, rate  = rrRate)
#     }else{
#       rrGenes[i] = 1#rgamma(1, ctrlRRShape, rate = ctrlRRrate)
#     }
#     
#     pVariantsGivenDisease1ByGene[i] = probVariantGivenDisease(rrGenes[i], afGenes[i])
#     pVariantsGivenNoDiseaseByGene[i] = probVariantGivenNotDisease(prevalence1, afGenes[i], pVariantsGivenDisease1ByGene[i])
#     
#     # if(isDisease == 1) {
#     #   print(paste("RR", rrGenes[i], "pVariantsGivenDisease1ByGene", pVariantsGivenDisease1ByGene[i], sep=" "))
#     # }
#     # print(pVariantsGivenDisease1ByGene[i])
#     
#     case1AlleleCountsByGene[i] = rbinom(1, nCase1, pVariantsGivenDisease1ByGene[i])
#     ctrl1AlleleCountsByGene[i] = rbinom(1, nCtrl1, pVariantsGivenNoDiseaseByGene[i])
#   }
#   
#   list(rrGenes = rrGenes, afGenes = afGenes, ctrl1AlleleCountsByGene = ctrl1AlleleCountsByGene, case1AlleleCountsByGene = case1AlleleCountsByGene,
#        pVariantsGivenDisease1ByGene = pVariantsGivenDisease1ByGene, pVariantsGivenNoDiseaseByGene = pVariantsGivenNoDiseaseByGene)
# }

fitFn = function(ctrl1AlleleCountsByGene, case1AlleleCountsByGene, prevalence1) {
  likelihoodFn = likelihoodUnivariate(ctrl1AlleleCountsByGene, case1AlleleCountsByGene, prevalence1 = prevalence1)
  results = list(ll = c(), par = c())
  minLLDiff = 1
  minLLThresholdCount = 5
  thresholdHitCount = 0
  nEpochs = 100
  for(i in 1:nEpochs) {
    piInitialGuess = rand(n = 1)[[1]]
    pDiseasesGivenVariantInitiailGuess = rand(n = 1)[[1]]
    fit = NULL
    tryCatch({
      fit = optim(par = list(pi1 = .00001, pDiseaseGivenVariant = pDiseasesGivenVariantInitiailGuess), fn = likelihoodFn,
                  control=list(fnscale=-1, max_iter = 2000))#,
      #method="L-BFGS-B", lower=list(pDiseasesGivenVariant = .00000001), upper=list(pDiseasesGivenVariant = .9999999999))
    }, error = function(e) {
      print(paste("Couldn't evaluate with parameters: ", piInitialGuess, pDiseasesGivenVariantInitiailGuess, sep = " "))
      fit = NULL
    }) # method="L-BFGS-B", lower=list(pDiseasesGivenVariant = .00000001), upper=list(pDiseasesGivenVariant = .9999999999)
    
    if(is.null(fit) || fit$value == 0 || fit$convergence != 0 || fit$par[["pi1"]] <= 0 || fit$par[["pi1"]] >= 1 || fit$par[["pDiseaseGivenVariant"]] <= 0 || fit$par[["pDiseaseGivenVariant"]] >= 1 ) {
      print("Failed to converge")
      print(fit)
      next
    }
    
    if(length(results$ll) == 0) {
      results$ll = append(results$ll, fit$value)
      results$par = append(results$par, data.frame(fit$par))
      next
    }
    
    maxPrevious =  max(results$ll)
    
    if(fit$value != maxPrevious && (fit$value - maxPrevious) > minLLDiff) {
      thresholdHitCount = 0
      print(fit)
      results$ll = append(results$ll, fit$value)
      results$par = append(results$par, data.frame(fit$par))
      next
    }
    
    thresholdHitCount = thresholdHitCount + 1
    print(fit)
    if(thresholdHitCount >= minLLThresholdCount) {
      print("DONE")
      break
    }
  }
  
  results
}

results = list(pDiseaseGivenVariant = c(), pi1 = c(), ll = c(), afShape = c(), afMean = c(), rrShape = c(), rrMean = c(), ctrlRRShape = c(), prevalence1 = c(), diseaseFraction = c())
for(rrShape in c(1, 10, 20, 50, 100)) {
  ctrlRRShape = 100
  for(prevalence1 in c(.01, .05, .1)) {
    for(rrMean in c(1, 2, 5, 10, 20, 50)) {
      for(afMean in c(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)) {
        for(afShape in c(1, 10, 20, 50, 100)) {
          for(diseaseFraction in c(.01, .05, .1, .2)) {
            print(paste("TESTING WITH:", "rrMean", rrMean, "rrShape", rrShape, "ctrlRRShape", ctrlRRShape, "afMean", afMean, "afShape", afShape, "diseaseFraction", diseaseFraction, "prevalence1", prevalence1))
            nGenes = 20000
            
            nCase1 = 1000
            nCtrl1 = 100000
            library("pracma")
            
            rrGenes = vector(length = nGenes)
            afGenes = vector(length = nGenes)
            pVariantsGivenDisease1ByGene = vector(length = nGenes)
            pVariantsGivenNoDiseaseByGene = vector(length = nGenes)
            case1AlleleCountsByGene = vector(length = nGenes)
            ctrl1AlleleCountsByGene = vector(length = nGenes)
            
            tryCatch({
              for(i in 1:nGenes){
                afGenes[i] = rgamma(1, afShape, rate  = afShape / afMean)
                
                if(i <= (nGenes*diseaseFraction)){
                  rrGenes[i] = rgamma(1, shape = rrShape, rate  = rrShape/rrMean)
                }else{
                  rrGenes[i] = rgamma(1, shape = ctrlRRShape, rate = ctrlRRShape)
                }
                
                pVariantsGivenDisease1ByGene[i] = probVariantGivenDisease(rrGenes[i], afGenes[i])
                pVariantsGivenNoDiseaseByGene[i] = probVariantGivenNotDisease(prevalence1, afGenes[i], pVariantsGivenDisease1ByGene[i])
                
                case1AlleleCountsByGene[i] = rbinom(1, nCase1, pVariantsGivenDisease1ByGene[i])
                ctrl1AlleleCountsByGene[i] = rbinom(1, nCtrl1, pVariantsGivenNoDiseaseByGene[i])
              }
              
              res = fitFn(ctrl1AlleleCountsByGene, case1AlleleCountsByGene, prevalence1 = prevalence1)
              maxIdx = which(res$ll == max(res$ll))
              maxLL = res$ll[maxIdx]
              maxPar = res$par[maxIdx]
              
              if(!is.null(maxPar)) {
                results$diseaseFraction = append(results$diseaseFraction, diseaseFraction)
                results$prevalence1 = append(results$prevalence1, prevalence1)
                
                results$ctrlRRShape = append(results$ctrlRRShape, ctrlRRShape)
                
                results$rrShape = append(results$rrShape, rrShape)
                results$rrMean = append(results$rrMean, rrMean)
                
                results$afShape = append(results$afShape, afShape)
                results$afMean = append(results$afMean, afMean)
                
                results$ll = append(results$ll, maxLL)
                results$pi1 = append(results$pi1, maxPar$fit.par[1])
                results$pDiseaseGivenVariant = append(results$pDiseaseGivenVariant, maxPar$fit.par[2])
              }
            }, error = function(e) {
              print(paste("Couldn't run with rrShape", rrShape, "because:", e, sep=" "))    
            })
          }
        }
      }
    }
  }
}

rgamma(1, shape = 100, rate = 100)
floor(xfit)
xfit<-floor(seq(min(0),max(10000),length=10000))
yfit = rgamma(xfit, shape = 1, rate = 1 / 20)
lines(xfit , y = yfit, col = "blue" , lty = 2 , lwd = 2 )
# compare fit to "true" parameter
# c(fit$par, loglik=tst(fit$par))
# expected_pDiseaseGivenVariant = mean(pVariantsGivenDisease1ByGene[1:(nGenes*disFrac)]*prevalence1/afGenes[1:(nGenes*disFrac)])
# true_param = list(pDiseaseGivenVariant = expected_pDiseaseGivenVariant, pi1=disFrac)
# c(unlist(true_param), loglik=tst(true_param))

# optim(par = list(pi1 = piInitialGuess, pDiseaseGivenVariant = .0001), fn = likelihoodFn, control=list(fnscale=-1, max_iter = 2000))
