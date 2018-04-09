import torch.nn as nn


class Net(nn.Module):
    """
    Input: 32*32*3
    conv1(3, 8, kernel_size=5) -> 8*28*28
    max_pool1(2, 2) -> 8*14*14
    conv2(8, 16, kernel_size=3) -> 16*12*12
    max_pool2(2, 2) -> 16*6*6
    conv3(16, 32, kernel_size=3) -> 32*4*4
    max_pool3(4, 4) -> 32*1*1

    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size = 7)
        self.max_pool = nn.