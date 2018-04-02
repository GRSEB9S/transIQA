import torch.nn


class Net(nn.Module):
    """
    Input: 32*32*1
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size = 7)
        self.max_pool = nn.