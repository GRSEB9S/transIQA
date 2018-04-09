import torch.nn as nn
import math
import torch.nn.functional as F
import torch


class Net(nn.Module):
    """
    Input: 32*32*3
    conv1(3, 8, kernel_size=5) -> 8*28*28
    max_pool1(2, 2) -> 8*14*14
    conv2(8, 16, kernel_size=3) -> 16*12*12
    max_pool2(2, 2) -> 16*6*6
    conv3(16, 32, kernel_size=3) -> 32*4*4
    max_pool3(4, 4) -> 32*1*1
    min_pool3(4, 4) -> 32*1*1

    concat(max_pool3 + min_pool3)
    fc1(64, 256) -> 256
    fc2(256, 1) -> 1

    """
    def __init__(self):
        super(Net, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=0),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
        )

        self.classifiers = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 1)
        )

        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.classifiers.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = self.features(x)

        x_min = - F.max_pool2d(-x, kernel_size=4, stride=4)
        x_max = F.max_pool2d(x, kernel_size=4, stride=4)
        x_min = x_min.view(x_min.size(0), -1)
        x_max = x_max.view(x_max.size(0), -1)
        x = torch.cat((x_min, x_max), 0)

        x = self.classifiers(x)

        return x
