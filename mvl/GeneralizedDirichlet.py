import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

from torch import tensor, autograd, Tensor
from pyro.distributions import DirichletMultinomial
import torch

# https://discuss.pytorch.org/t/computing-the-hessian-matrix-correctness-and-performance/53624
def compute_hessian(grads, params):
    H = []
    for i, (grad, p) in enumerate(zip(grads, params)):
        grad = grad.reshape(-1)
        d = len(grad)
        dg = torch.zeros((d, d))

        for j, g in enumerate(grad):
            g2 = autograd.grad(g, p, create_graph=True)[0].view(-1)
            dg[j] = g2

        H.append(dg)

    return H

# based on MGLM
def dgdirmn(Y, alpha, beta): 
    assert beta.shape == alpha.shape

    assert alpha.shape[0] == Y.shape[1] - 1
    alpha = alpha.expand((Y.shape[0], alpha.shape[0]))#     d <- ncol(Y)
    beta = beta.expand((Y.shape[0], beta.shape[0]))#     d <- ncol(Y)

    m = Y.sum(1) #m
    Yrev = Y.T.flip(0) 
    YrevCumsum = Yrev.cumsum(0) 
    z = YrevCumsum.flip(0).T

    n1 = torch.lgamma(m + 1)
    n2 = (torch.lgamma(Y[:, :-1] + alpha)).sum(1)
    n3 = (torch.lgamma(z[:, 1:] + beta)).sum(1)
    n4 = (torch.lgamma(alpha + beta)).sum(1)

    numerator = n1 + n2 + n3 + n4
    
    d1 = torch.lgamma(Y + 1).sum(1)
    d2 = torch.lgamma(alpha).sum(1)
    d3 = torch.lgamma(beta).sum(1)
    d4 = torch.lgamma(alpha + beta + z[:, :-1]).sum(1)

    denominator = d1 + d2 + d3 + d4

    return numerator - denominator

##============================================================## 
## Generate GDM data 
##============================================================##
#' @rdname gdirmn 
#' 
#' @param n the number of random vectors to generate.  When \code{size} is a scalar and \code{alpha} is a vector, 
#' must specify \code{n}.  When \code{size} is a vector and \code{alpha} is a matrix, \code{n} is optional.
#' The default value of \code{n} is the length of \code{size}. If given, \code{n} should be equal to 
#' the length of \code{size}.
#' @param size a number or vector specifying the total number of objects that are put
#' into d categories in the generalized Dirichlet multinomial distribution.
#' 
#' @export rgdirmn 
# rgdirmn <- function(n, size, alpha, beta) {
    
#     if (length(alpha) != length(beta)) 
#         stop("The size of alpha and beta should match.")
    
#     if (is.vector(alpha) && missing(n)) 
#         stop("When alpha and beta are vectors, must give n.")
    
#     if (is.vector(alpha)) {
#         alpha <- t(matrix(alpha, length(alpha), n))
#         beta <- t(matrix(beta, length(beta), n))
#         if (length(size) == 1) {
#             size <- rep(size, n)
#         } else {
#             stop("The length of size variable doesn't match with alpha or beta")
#         }
#     } else {
#         if (missing(n)) {
#             n <- dim(alpha)[1]
#         } else if (length(n) != 1) {
#             stop("n should be a scalar.")
#         } else if (n != dim(alpha)[1]) 
#             stop("The sizes of the input alpha don't match with n")
#     }
    
#     k <- dim(alpha)[2]
#     if (k < 1) {
#         stop("The multivariate response data need to have more than one category.")
#     }
    
#     if (!is.vector(size)) {
#         stop("n must be a scalar, or a column vector matches with \n\t\t\tthe number of rows of alpha")
#     } else if (length(size) != n) 
#         stop("The length of size should match with n")
    
#     ##----------------------------------------## 
#     ## Generate data
#     ##----------------------------------------## 
    
#     rdm <- matrix(0, n, (k + 1))
#     rdm[, 1] = size
    
#     for (i in 1:k) {
#         rdm[, c(i, i + 1)] <- rdirmn(size = rdm[, i], alpha = cbind(alpha[, i], beta[, 
#             i]), n)
#     }
    
#     return(rdm)
# }
# size is the marginal count
def rgdirmn(size: Tensor, alphas: Tensor, betas: Tensor):
    assert len(alphas.shape) > 0 and len(betas.shape) > 0 #is list
    assert alphas.shape == betas.shape

    # number of random vectors to generate
    n = alphas.shape[0]
    k = alphas.shape[1]
    # print("k", k)
    # print("n", n)
    assert k >= 1
    
    assert size.shape[0] == n    
    res = torch.zeros(n, k)
    res[:, 0] = size[0]
    # print("res", res)
    # print("res", res)
    # count_col = n.expand(n, 2)
    concentrations = torch.stack((tensor(alphas), tensor(betas))).T
    for k_idx in range(0, k - 1):
        # if torch supported inhomogenous total_count, this would work:
        #dm = DirichletMultinomial(concentration = concentration, total_count = res[:, i], validate_args=False)
        for n_idx in range(n):
            concentration = concentrations[k_idx, n_idx]
            # print(f"at k=={k_idx}, n_idx={n_idx}, concentration={concentration}")
            if res[n_idx, k_idx] == 0:
                res[n_idx, k_idx:k_idx+2] = tensor([0., 0.])
            else:
                dm = DirichletMultinomial(concentration = concentration, total_count = res[n_idx, k_idx])
                sample = dm.sample()
                res[n_idx, k_idx:k_idx+2] = sample

                # print("sample", sample)
                # print("sample", sample)
                # print("res[:, i:i+1]", res[:, i:i+2])
    
    return res

# Y = tensor([
#     [0, 1, 19],
#     [9, 11, 0],
#     [0, 0, 20],
#     [1, 19, 0],
#     [0, 16, 4],
#     [2, 14, 4],
#     [1, 2, 17],
#     [19, 1, 0],
#     [2, 16, 2],
#     [0, 20, 0]
# ]).type(torch.float32)

# alpha = tensor([0.2, 0.5]).requires_grad_(True)
# beta = tensor([0.7, 0.4]).requires_grad_(True)

# loss = -dgdirmn(Y, alpha, beta)

# loss.sum().backward()
# print(alpha.grad)
# print(beta.grad)

# print(compute_hessian(alpha.grad, alpha))