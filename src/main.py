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
import gc
import numpy as np

parser = argparse.ArgumentParser(description='TransIQA')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10)

cuda = 1 and torch.cuda.is_available()
log_interval = 200
epochs = 75000000 # for 5W images, each images 30 patches, each patch 10 times
per_epoch = 1500000 # 50 times for one image
per_epoch = 2
lr = 0.00001
momentum = 0.5
txt_input = './data/face_score_generated_dlib.txt'
batch_size = 32
num_workers = 4
num_faces = 10000 #7G ROM for 10000 28*28*3 numpy array

model = model.Net()
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

#face_dataset = dataset.FaceScoreDataset(image_list=txt_input,
#                                        transform = transforms.Compose([
#                                            dataset.RandomCrop(32),
#                                            dataset.ToTensor()
#
#                                         ]))
#debug for Dataset
debug=0
if debug:
    print('FaceScoreDataset Initial: OK!')
    exit(0)

#for i in range(len(face_dataset)):
#    sample = face_dataset[i]
#
#    print(i, sample['image'].shape, sample['score'])
#    tools.show_image(sample['image'], sample['score'])

#dataloader = DataLoader(face_dataset, batch_size=batch_size,
#                        shuffle=True, num_workers=num_workers)

# debug code
#debug = 0
#if debug:
#    for i_batch, sample_batched in enumerate(dataloader):
#        print(i_batch, sample_batched['image'].size(),
#              sample_batched['score'].size())


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


def test():
    #model.test()

    face_dataset = dataset.FaceScoreDataset(image_list=txt_input,
                                            train = False)

    images = face_dataset.images
    scores = face_dataset.scores
    outputs = []

    for i in range(len(images)):

        image = images[i]
        height = image.shape[0]
        width = image.shape[1]

        patches = []
        for i in range(30): # num of small patch
            top = np.random.randint(0, height - 32)
            left = np.random.randint(0, width - 32)
            patches.append(image[top:top+32, left:left+32, :].transpose((2, 0, 1)))

        patches = np.array(patches)
        #debug
        debug=0
        if debug:
            print(patches)
            print(patches.shape)
        patches = torch.from_numpy(patches)

        if cuda:
            patches = patches.cuda()
        patches = Variable(patches)

        output = model(patches).data
        output = sum(output) / 30
        outputs.append(output)

    loss = np.mean(np.abs(np.array(outputs - scores)))
    print('Testing Loss:{:.6f}'.format(loss))



for epoch in range(1, epochs + 1):

    # reload dataset
    if epoch % per_epoch == 1:
        if epoch > 1:
            del face_dataset.images #IMPORTANT: delete list type variables is useful
            del dataloader
            gc.collect()
            test()
        face_dataset = dataset.FaceScoreDataset(image_list=txt_input,
                                                transform = transforms.Compose([
                                                    dataset.RandomCrop(32),
                                                    dataset.ToTensor()
                                                ]),
                                                num_faces = num_faces)
        dataloader = DataLoader(face_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

    train(epoch)
