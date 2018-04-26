import torch
import numpy as np
from model import Net_deep, ft12, ft2
import tools
import argparse
import torch.nn.functional as F
import copy
from torch.optim.lr_scheduler import StepLR


'''
Fine-tune on LIVE/TID2013 dataset

Refer main.py
1. load_model
2. get dataloader
3. train
4. test
'''

parser = argparse.ArgumentParser(description='fine-tune exist model on LIVE/TID2013.')

parser.add_argument('--dataset', type=str, default='live',
                    help='[live]live or tid2013 for fine-tuning')
parser.add_argument('--load_model', type=str, default='./model/cuda_True_epoch_550',
                    help='model path to fine-tune from')
parser.add_argument('--epochs', type=int, default=500,
                    help='[200]total epochs of training')
parser.add_argument('--batch_size', type=int, default=32,
                    help='[32]batch_size')
parser.add_argument('--num_workers', type=int, default=4,
                    help='[4]num_workers for iterating traning dataset')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='[1e-5] learning rate')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='[adam] optimizer type')
parser.add_argument('--train_loss', type=str, default='mae',
                    help='[mae] training loss function')
parser.add_argument('--test_loss', type=str, default='mae',
                    help='[mae] testing loss function')
parser.add_argument('--data_log', type=str, default='',
                    help='['']path to write data for visualization')
parser.add_argument('--model_reload', type=str, default='',
                    help='['']path to reload model')
parser.add_argument('--epoch_reload', type=int, default=0,
                    help='[0]epoch when reloaded model finished')
parser.add_argument('--model_save', type=str, default='',
                    help='['']model saving path')
parser.add_argument('--model_epoch', type=int, default=100,
                    help='[100]epochs for saving the best model')
parser.add_argument('--mode', type=str, default='ft',
                    help='[ft]ft:ft on deepnet, ft12, ft2')
args = parser.parse_args()

dataset = args.dataset
epochs = args.epochs
batch_size = args.batch_size
num_workers = args.num_workers
optimizer = args.optimizer
lr = args.lr
train_loss = args.train_loss
test_loss = args.test_loss
data_log = args.data_log
model_reload = args.model_reload
epoch_reload = args.epoch_reload
model_save = args.model_save
model_epoch = args.model_epoch
mode = args.mode
write_data = True if data_log != '' \
    else False
reload = True if model_reload != '' and epoch_reload != 0 \
    else False
model_path = args.load_model if not reload \
    else args.model_reload
save_model = True if model_save != '' \
    else False

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
dtype = torch.float

if dataset == 'tid2013':
    live_train = './data/tid2013_generator/ft_tid2013_train.txt'
    live_test = './data/tid2013_generator/ft_tid2013_test.txt'
else:
    live_train = './data/live_generator/ft_live_train.txt'
    live_test = './data/live_generator/ft_live_test.txt'

# if model_path == '', train from scratch
if model_path != '':
    model = torch.load(model_path)
    if mode == 'ft12':
        model = ft12(model)
    elif mode == 'ft2':
        model = ft2(model)
else:
    model = Net_deep()
    if mode == 'ft12':
        model = ft12(model)
    elif mode == 'ft2':
        model = ft2(model)

model.to(device=device, dtype=dtype)
if save_model:
    best_model = {'model': None,
                  'epoch': -1,
                  'loss': -1,
                  'lcc': 0.95,
                  'srocc': 0.95,
                  'new': False}

if mode == 'ft12':
    optimizer = torch.optim.Adam([
        {'params': model.features.parameters()},
        {'params': model.classifiers.parameters()},
        {'params': model.logistic.parameters(), 'lr': 6e4*lr}
    ], lr=lr, betas=(0.9, 0.99))
elif mode == 'ft2':
    optimizer = torch.optim.Adam([
        {'params': model.classifiers.parameters(),},
        {'params': model.logistic.parameters(), 'lr': 6e4*lr}
    ], lr=lr, betas=(0.9, 0.99))
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

# for one iter
scheduler = StepLR(optimizer, step_size=25, gamma=0.7)

live_dataset = tools.get_live_dataset(live_train=live_train,
                                      live_test=live_test)
data_loader = tools.get_dataloader(live_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=num_workers)


def train(epoch=1):
    model.train()
    patches_per_image = 30

    # for one image, sample 30 times
    for i in range(patches_per_image):
        for batch_idx, sample_batched in enumerate(data_loader):
            image = sample_batched['image'].to(device=device, dtype=dtype)
            score = sample_batched['score'].to(device=device, dtype=dtype)

            optimizer.zero_grad()
            output = model(image)
            if train_loss == 'mse':
                loss = F.mse_loss(output, score)
            elif train_loss == 'mae':
                loss = F.l1_loss(output, score)
            else: exit(0)
            loss.backward()
            optimizer.step()

            # debug
            if str(loss) == 'nan':
                print(output.size())

    tools.log_print('Epoch_{} Loss: {:.2f}'.format(
        epoch, loss.detach()))
    lcc, srocc = tools.evaluate_on_metric(output, score)

    if write_data:
        with open(data_log, 'a') as f:

            line = 'train epoch:{} percent:{:.6f} loss:{:.4f} lcc:{:.4f} srocc:{:.4f}\n'
            line = line.format(epoch,
                               0.,
                               loss.detach(),
                               lcc,
                               srocc)
            f.write(line)


def test(epoch):
    model.eval()

    images = live_dataset.test_images
    scores = live_dataset.test_scores
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
        patches = torch.from_numpy(patches).to(device)

        output = model(patches).detach()
        output = sum(output) / 30
        outputs.append(output)

    if test_loss == 'mse':
        loss = np.mean((np.array(outputs - scores)) ** 2)
    elif test_loss == 'mae':
        loss = np.mean(np.abs(np.array(outputs - scores)))
    tools.log_print('TESTING LOSS:{:.6f}'.format(loss))
    lcc, srocc = tools.evaluate_on_metric(outputs, scores)

    if write_data:
        with open(data_log, 'a') as f:
            f.write('test loss:{:.4f} lcc:{:.4f} srocc:{:.4f}\n'.format(loss, lcc, srocc))

    # save best model
    if save_model and srocc > best_model['srocc'] and lcc > best_model['lcc']:
        best_model['model'] = copy.deepcopy(model)
        best_model['epoch'] = epoch
        best_model['loss'] = loss.detach()
        best_model['lcc'] = lcc
        best_model['srocc'] = srocc
        # update 'new' buffer
        best_model['new'] = True


print('Logging data info to: {}'.format(data_log))
print(args)
print(model)
if write_data:
    with open(data_log, 'a') as f:
        f.write(str(args) + '\n')
        f.write(str(model) + '\n')

for i in range(epochs):
    epoch = i + 1 + epoch_reload
    scheduler.step()

    test(epoch)
    train(epoch)

    if save_model and i % model_epoch == 0 and best_model['new'] == True:
        path = model_save + '_{}_{:.4f}_{:.4f}'.format(
            best_model['epoch'], best_model['lcc'], best_model['srocc'])
        tools.log_print('Saving model:{}'.format(path))
        torch.save(best_model['model'].to(torch.device("cpu")),
                   path)
        # close buffer
        best_model['new'] = False
