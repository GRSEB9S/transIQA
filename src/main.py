import tools
from model import Net_deep
import dataset
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import gc
import numpy as np

parser = argparse.ArgumentParser(description='TransIQA')

parser.add_argument('--limited', action="store_true",
                    help='run with small datasets limited to 8G memory usage')
parser.add_argument('--no_cuda', action="store_true",
                    help='cuda disable switch')

parser.add_argument('--log_interval', type=int, default=50,
                    help='[50]percentage of one epoch for loss output')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='[1e-4]learning rate')
parser.add_argument('--txt_input', type=str, default='./data/face_score_generated_dlib.txt',
                    help='[./data/face_score_generated_dlib.txt]input image path for training and validation')
parser.add_argument('--batch_size', type=int, default=64,
                    help='[64]input batch size for training')
parser.add_argument('--num_workers', type=int, default=4,
                    help='[4]workers for getting pictures')
parser.add_argument('--epochs', type=int, default=50,
                    help='[50]total training epoch')
parser.add_argument('--model_epoch', type=int, default=50,
                    help='[50]epoch for saving model')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='[adm]choose optimizer')
parser.add_argument('--train_loss', type=str, default='mse',
                    help='[mse]define training loss used')
parser.add_argument('--test_loss', type=str, default='mse',
                    help='[mse]define testing loss used')
parser.add_argument('--data_log', type=str, default='',
                    help='['']path to write data for visualization')
parser.add_argument('--data_log_per_epoch', type=int, default='100',
                    help='[100]per epoch for one training data log')
parser.add_argument('--reload_model', type=str, default='',
                    help='['']path for model to continue')
parser.add_argument('--reload_epoch', type=int, default=0,
                    help='[0]epoch of the model to continue')

args = parser.parse_args()

cuda = ~args.no_cuda and torch.cuda.is_available()
log_interval = args.log_interval
epochs = args.epochs # for 5W images, each images 30 patches, each patch 10 times
lr = args.lr
txt_input = args.txt_input
batch_size = args.batch_size
num_workers = args.num_workers
limited = args.limited
model_epoch = args.model_epoch
data_log = args.data_log
reload_model = args.reload_model
reload_epoch = args.reload_epoch
if data_log != '':
    write_data = True
else: write_data = False
data_log_per_epoch = args.data_log_per_epoch
if reload_model != '' and reload_epoch != 0:
    reload = True



if limited:
    num_faces = 2000 #7G ROM for 10000 28*28*3 numpy array
else:
    num_faces = 10000

tools.log_print('{} faces/split'.format(num_faces))

if reload:
    model = torch.load(reload_model)
else:
    model = Net_deep()
if cuda:
    model.cuda()

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.8)
else: exit(0)


def train(epoch=1, limited=True):
    model.train()

    num_split = int(50000 / num_faces)
    patch_per_face = 30
    num_total_patch = 50000 * patch_per_face
    log_patch = int(num_total_patch / log_interval)
    data_log_per_patch = num_total_patch // data_log_per_epoch
    data_log_time = 0

    # for one train, five dataloader
    for i in range(num_split):
        face_dataset = tools.get_dataset(limited=True, image_list = txt_input)

        # one dataset, get 30 dataloader
        for j in range(patch_per_face):
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
                if args.train_loss == 'mse':
                    loss = F.mse_loss(output, score)
                elif args.train_loss == 'mae':
                    loss = F.l1_loss(output, score)
                else: exit(0)
                loss.backward()
                optimizer.step()
                num_patches = i*patch_per_face*num_faces + j*num_faces + batch_idx*batch_size

                if write_data and data_log_time * data_log_per_patch < num_patches:
                    with open(data_log, 'a') as f:

                        lcc, srocc = tools.evaluate_on_metric(output, score, log=False)
                        line= 'train epoch:{} percent:{:.6f} loss:{:.4f} lcc:{:.4f} srocc:{:.4f}\n'.format(epoch,
                                                                                                           num_patches/num_total_patch,
                                                                                                           loss.data[0],
                                                                                                           lcc,
                                                                                                           srocc)
                        f.write(line)
                        data_log_time += 1

                if num_patches % log_patch == 0:
                    tools.log_print('Epoch_{} Split_{} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
                        epoch, i+1, num_patches, num_total_patch,
                                100. * num_patches / num_total_patch, loss.data[0]))
                    tools.evaluate_on_metric(output, score)

        # restore memory for next
        del face_dataset.images
        gc.collect()

        test(limited=limited)


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


    if args.test_loss == 'mse':
        loss = np.mean((np.array(outputs - scores)) ** 2)
    elif args.test_loss == 'mae':
        loss = np.mean(np.abs(np.array(outputs - scores)))
    tools.log_print('TESTING LOSS:{:.6f}'.format(loss))
    lcc, srocc = tools.evaluate_on_metric(outputs, scores)
    if write_data:
        with open(data_log, 'a') as f:
            f.write('test loss:{:.4f} lcc:{:.4f} srocc:{:.4f}\n'.format(loss, lcc, srocc))

    del face_dataset.images
    gc.collect()


print('Logging data info to: {}'.format(data_log))
print(args)
print(model)
if write_data:
    with open(data_log, 'a') as f:
        f.write(str(args) + '\n')
        f.write(str(model) + '\n')

for epoch in range(1, epochs+1):

    if reload:
        epoch += reload_epoch
    tools.log_print('Epoch: {}'.format(epoch))

    #if epoch % per_epoch == 1:
    #    tools.log_print('Testing')
    #    test(limited=limited)

    train(epoch, limited=args.limited)

    if epoch % model_epoch == 0:
        tools.save_model(model=model, model_name='cuda_' + str(cuda), epoch=epoch)
