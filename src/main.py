import tools
import model
import dataset
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser(description='TransIQA')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=10)


txt_input = './data/image_live_2.txt'

model = model.Net()
if torch.cuda.is_available():
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

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

dataloader = DataLoader(face_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['score'].size())