import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

label = np.array([1., 2., 3.])
hypo = np.array([0., 0., 0.])
np_loss = np.mean((np.array(label - hypo)) ** 2)

label = torch.from_numpy(label)
hypo = torch.from_numpy(hypo)

print('Variable.numpy():{}'.format(label.numpy()))

label = Variable(label)
hypo = Variable(hypo)

mse_loss = F.mse_loss(label, hypo)


print(mse_loss.data)
print('np_loss:{}'.format(np_loss))

