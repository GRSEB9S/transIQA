import tools
import model
import dataset
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser(description='TransIQA')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10)

cuda = torch.cuda.is_available()
log_interval = 100
epochs = 100
lr = 0.0001
momentum = 0.5
txt_input = './data/image_score_generated.txt'
batch_size = 64

model = model.Net()
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#crop = dataset.RandomCrop(32)
face_dataset = dataset.FaceScoreDataset(image_list=txt_input,
                                        transform = transforms.Compose([
                                            dataset.RandomCrop(32),
                                            dataset.ToTensor()
                                        ]))

#for i in range(len(face_dataset)):
#    sample = face_dataset[i]
#
#    print(i, sample['image'].shape, sample['score'])
#    tools.show_image(sample['image'], sample['score'])

dataloader = DataLoader(face_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=1)# glymur not support multi-processing

# debug code
debug = False
if debug:
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['score'].size())

def train(epoch=1):
    model.train()
    for batch_idx, sample_batched in enumerate(dataloader):
        image = sample_batched['image']
        score = sample_batched['score']
        if cuda:
            image, score = image.cuda(), score.cuda()
        image, score = Variable(image), Variable(score)

        #debug
        debug = False
        if debug:
            print(type(sample_batched['image']))
            print(type(image))
            print(score)
            exit(0)

        optimizer.zero_grad()
        output = model(image)
        loss = F.l1_loss(output, score)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.data[0]))

for epoch in range(1, epochs + 1):
    train(epoch)
