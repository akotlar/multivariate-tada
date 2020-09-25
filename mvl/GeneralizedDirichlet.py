import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

from torch import tensor
import torch

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

Y = tensor([
    [0, 1, 19],
    [9, 11, 0],
    [0, 0, 20],
    [1, 19, 0],
    [0, 16, 4],
    [2, 14, 4],
    [1, 2, 17],
    [19, 1, 0],
    [2, 16, 2],
    [0, 20, 0]
]).type(torch.float32)

alpha = tensor([0.2, 0.5]).requires_grad_(True)
beta = tensor([0.7, 0.4]).requires_grad_(True)

loss = -dgdirmn(Y, alpha, beta)

loss.sum().backward()
print(alpha.grad)
print(beta.grad)