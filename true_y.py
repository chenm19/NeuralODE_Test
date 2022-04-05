import math
import os
import argparse
import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_false')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()
# print(args.viz)
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[-1.]]).to(device)
t = torch.linspace(1., 10., args.data_size).to(device)
class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.tensor(t*y**2)
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
plt.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0], 'g-')
plt.grid()
plt.show()
y = -2/(t**2 +1)
plt.plot(t.cpu().numpy(), y.cpu().numpy(), 'g-')
plt.show()