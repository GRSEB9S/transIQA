import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np
from collections import OrderedDict


class Net_deep(nn.Module):
    """
    from: A DEEP NEURAL NETWORK FOR IMAGE QUALITY ASSESSMENT
    conv3-32, conv3-32, maxpool,
    conv3-64, conv3-64, maxpool,
    conv3-128, conv3-128, maxpool,
    conv3-256, conv3-256, maxpool,
    conv3-512, conv3-512, maxpool,
    FC-512, FC-1.
    {
    Input: 32*32*3

    conv1a(3, 32, kernel_size=3, padding=1) -> 32*32*32
    ReLU(inplace=True) -> 32*32*32
    conv1b(32, 32, kernel_size=3, padding=1) -> 32*32*32
    ReLU(inplace=True) -> 32*116*16
    max_pool1(kernel_size=2, stride=2) -> 32*16*16

    conv2a(32, 64, kernel_size=3, padding=1) -> 64*16*16
    ReLU(inplace=True) -> 32*32*32
    conv2b(64, 64, kernel_size=3, padding=1) -> 64*16*16
    ReLU(inplace=True) -> 64*16*16
    max_pool2(2, 2) -> 64*8*8

    conv3a(64, 128, kernel_size=3, padding=1) -> 128*8*8
    ReLU(inplace=True) -> 128*8*8
    conv3b(128, 128, kernel_size=3, padding=1) -> 128*8*8
    ReLU(inplace=True) -> 128*8*8
    max_pool3(2, 2) -> 128*4*4

    conv4a(128, 256, kernel_size=3, padding=1) -> 256*4*4
    ReLU(inplace=True) -> 256*4*4
    conv4b(256, 256, kernel_size=3, padding=1) -> 256*4*4
    ReLU(inplace=True) -> 256*4*4
    max_pool4(2, 2) -> 256*2*2

    conv5a(256, 512, kernel_size=3, padding=1) -> 512*2*2
    ReLU(inplace=True) -> 512*2*2
    conv5b(512, 512, kernel_size=3, padding=1) -> 512*2*2
    ReLU(inplace=True) -> 512*2*2
    max_pool5(2, 2) -> 512*1*1

    view(-1)

    fc1(512, 512) -> 512
    Dropout(p=0.5, inplace=True)
    fc2(512, 1) -> 1
    }

    """
    def __init__(self):
        super(Net_deep, self).__init__()

        self.features = nn.Sequential(

            #split1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #split2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #split3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #split4
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #split5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifiers = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(512, 1)
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.classifiers.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight,
                    gain=nn.init.calculate_gain('relu')
                )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifiers(x)

        return x


class Logistic(nn.Module):

    def __init__(self):
        super(Logistic, self).__init__()
        self.x0 = nn.Parameter(torch.from_numpy(np.array([0.2], dtype=np.float32)))
        self.k = nn.Parameter(torch.from_numpy(np.array([0.2], dtype=np.float32)))
        self.L = nn.Parameter(torch.from_numpy(np.array([0.2], dtype=np.float32)))

    def forward(self, x):
        x0 = self.x0.expand_as(x)
        k = self.k.expand_as(x)
        L = self.L.expand_as(x)
        return L / (1. + torch.exp((-k) * (x - x0)))


def ft12(model):
    """
    input Net_deep and
    add fc3 + logistic to classifiers
    all the params requires grad
    :param model:
    :return:
    """

    model.classifiers._modules['2'] = nn.Linear(512, 512)
    model.classifiers.add_module('dropout', nn.Dropout(p=0.5, inplace=True))
    model.classifiers.add_module('fc3', nn.Linear(512, 1))
    model.add_module('logistic', Logistic())

    return model


def ft2(model):
    """
    input Net_deep and output model
    add classifiers with fc3 and logistic
    only backward on classifiers
    :param model:
    :return:
    """

    model = ft12(model)
    for param in model.features.parameters():
        param.requires_grad = False

    return model