import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])

import torch
from torch import autograd

x = torch.tensor([2., 1.]).requires_grad_(True)
y = 5*x**4 + 3*x**3 + 7*x**2 + 9*x - 5

deriv_x = 5*4*x**3 + 3*3*x**2 + 7*2*x + 9
second_deriv_x = 5*4*3*x**2 + 3*3*2*x + 7*2

loss = y.sum()
deriv = autograd.grad(loss, x, create_graph=True)[0]
loss = deriv.sum()
second_deriv = autograd.grad(loss, x)
print("deriv", deriv, "secondderiv", second_deriv)
print("deriv_x", deriv_x, "second_deriv_x", second_deriv_x)