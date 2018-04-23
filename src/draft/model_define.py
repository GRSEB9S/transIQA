import torch
from torch.autograd import Variable
import math
from torch import nn
import numpy as np
import torch.nn.functional as F

'''
define logistic function
L / (1 + exp(-k(x-x0)))
'''


class Logistic(nn.Module):

    def __init__(self):
        super(Logistic, self).__init__()
        self.x0 = nn.Parameter(torch.from_numpy(np.array([1.])))
        self.k = nn.Parameter(torch.from_numpy(np.array([2.])))
        self.L = nn.Parameter(torch.from_numpy(np.array([3.])))

    def forward(self, x):
        x0 = self.x0.expand_as(x)
        k = self.k.expand_as(x)
        L = self.L.expand_as(x)
        return L / (1. + torch.exp((-k) * (x - x0)))


def grad_check():

    module = Logistic()
    x = np.array([1., 2., 3.])
    y = logistic(x)
    x = Variable(torch.from_numpy(x))
    lr = 1

    for i in range(5000):
        module.zero_grad()
        out = module(x)
        loss = F.mse_loss(out, Variable(torch.from_numpy(np.array(y))))
        loss.backward()

        for param in module.parameters():
            param.data -= lr * param.grad.data

        print(loss)

    print([param.data for param in module.parameters()])


def logistic(input):
    out = [2. / (1 + math.e**(-(x))) for x in input]
    return out

grad_check()