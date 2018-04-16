import tools
import model
import dataset
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import gc
import numpy as np

parser = argparse.ArgumentParser(description='TransIQA')
parser.add_argument('--limited', type=bool, default=True,
                    help='run with small datasets limited to 8G memory usage')
parser.add_argument('--cuda', type=bool, default=True,
                    help='cuda enable switch')
parser.add_argument('--log_interval', type=int, default=10,
                    help='percentage of one epoch for loss output')
parser.add_argument('--per_epoch', type=int, default=2,
                    help='validation output control')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5,
                    help='momentum')
parser.add_argument('--txt_input', type=str, default='./data/face_score_generated_dlib.txt',
                    help='input image path for training and validation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='workers for getting pictures')
parser.add_argument('--epochs', type=int, default=10)

args = parser.parse_args()

cuda = args.cuda and torch.cuda.is_available()
log_interval = args.log_interval
epochs = args.epochs # for 5W images, each images 30 patches, each patch 10 times
per_epoch = args.per_epoch
lr = args.lr
momentum = args.momentum
txt_input = args.txt_input
batch_size = args.batch_size
num_workers = args.num_workers

if args.limited == True:
    num_faces = 2000 #7G ROM for 10000 28*28*3 numpy array
else:
    num_faces = 10000

model = model.Net()
if cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

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


def train(epoch=1, limited=True):
    model.train()

    num_split = int(50000 / num_faces)
    patch_per_face = 30
    num_total_patch = 50000 * patch_per_face
    log_patch = int(num_total_patch / log_interval)

    # for one train, five dataloader
    for i in range(num_split):
        face_dataset = tools.get_dataset(limited=True, image_list = txt_input)

        # one dataset, get 30 dataloader
        for j in range(patch_per_face):
            #print('Training: epoch_{}, split_{}/{}, patch_{}/{}'.format(epoch,
            #                                                            i+1, num_split,
            #                                                            j+1, patch_per_face))
            dataloader = tools.get_dataloader(face_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
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
                loss = F.mse_loss(output, score)
                loss.backward()
                optimizer.step()

                num_patches = i*patch_per_face*num_faces + j*num_faces + batch_idx*batch_size
                if num_patches % log_patch == 0:
                    print('Train Epoch: {} Split: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i+1, num_patches, num_total_patch,
                                100. * num_patches / num_total_patch, loss.data[0]))
                    tools.evaluate_on_metric(output, score)


        # restore memory for next
        del face_dataset.images
        gc.collect()


def test(limited=True):
    #model.test()

    face_dataset = tools.get_dataset(limited=limited, train=False, image_list=txt_input)

    images = face_dataset.images
    scores = face_dataset.scores
    #debug
    debug=0
    if debug:
        print(scores)
        exit(0)

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

    loss = np.mean((np.array(outputs - scores)) ** 2)
    print('Testing Loss:{:.6f}'.format(loss))
    tools.evaluate_on_metric(outputs, scores)

    del face_dataset.images
    gc.collect()



for epoch in range(1, epochs + 1):

    print('epoch: {}'.format(epoch))

    if epoch % per_epoch == 1:
        print('Testing')
        test(limited=args.limited)

    train(epoch, limited=args.limited)
