import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import dataset
from scipy import stats
import time
import os.path as osp
from torch import Tensor

start = time.time()


def read_txt(image_path):

    images = [line.rstrip('\n').split()[0] for line in open(image_path)]
    scores = [line.rstrip('\n').split()[1] for line in open(image_path)]

    return images, scores

def show_image(image, title='default', commit=''):
    plt.title(str(title))
    plt.imshow(image)
    if commit != '':
        print(commit)
    plt.pause(0.01)
    plt.show()

def show_image_depth(image_path):
    image = io.imread(image_path)
    path = ''
    if len(image.shape)!=3:
        path = image_path
        print(path, image.shape)

    return path


def prepare_faces(scale = 1.2):
    import dlib

    image_list = './data/image_score_generated_dlib.txt'
    output_root = '../dataset/transIQA/faces'
    output_file = 'face_score_generated_dlib.txt'

    dlib_model_path = './model/mmod_human_face_detector.dat'

    images = [line.rstrip('\n').split()[0] for line in open(image_list)]
    scores = [line.rstrip('\n').split()[1] for line in open(image_list)]
    face_detector = dlib.cnn_face_detection_model_v1(dlib_model_path)
    faces = []
    face_scores = []

    print('Datasets reading and Face detecting')
    # prepare faces
    # detect faces and save as images file

    # read images and detect faces
    pristine_images = []
    num_pristine = int(len(images) / 21)
    for i in range(num_pristine):
        pristine_images.append(images[i * 21])

    # detect faces for one pristine
    for i in range(len(pristine_images)):
        print('Reading images: %s'%pristine_images[i])
        image_name = osp.splitext(osp.split(pristine_images[i])[1])[0]

        image = cv2.imread(pristine_images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = face_detector(image, 1)

        # get face locations
        face_locations = []
        for _, d in enumerate(dets):
            if d.confidence > 0.5:
                left, right, top, bottom = d.rect.left(), d.rect.right(), d.rect.top(), d.rect.bottom()
                height = bottom - top
                width = right - left
                central = (int((top+bottom)*0.5), int((left+right)*0.5))

                #scale image
                top = max(int(central[0] - height * scale * 0.5), 0)
                bottom = min(int(central[0] + height * scale * 0.5), image.shape[0])
                left = max(int(central[1] - width * scale * 0.5), 0)
                right = min(int(central[1] + width * scale * 0.5), image.shape[1])

                face_locations.append([top, bottom, left, right])

        # crop exist datasets
        if len(face_locations) > 0:
            for j in range(4*5+1):
                image = cv2.imread(images[i*21 + j])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                score = scores[i*21 + j]
                face_num = 0
                for location in face_locations:
                    #suffix

                    face_num += 1
                    suffix = osp.split(osp.split(images[i*21 + j])[0])[1] + '_' + str(face_num)

                    image_path = output_root + '/' + image_name + '_' + suffix + '.npy'
                    #print('saving numpy image: %s'%image_path)
                    np.save(image_path, image[location[0]:location[1], location[2]:location[3], :])

                    faces.append(image_path)
                    face_scores.append(score)

                    #debug
                    debug=0
                    if debug:
                        plt.imshow(np.load(image_path))
                        plt.title(str(score))
                        plt.show()

        print('%d in %d'%(i, len(pristine_images)))

    with open('../dataset/transIQA' + '/' + output_file, 'w') as f:
        for i in range(len(faces)):
            f.write(faces[i] + ' ' +str(face_scores[i]) + '\n')


def get_dataset(limited=True,
                train=True,
                image_list='',
                transform=transforms.Compose([
                    dataset.RandomCrop(32),
                    dataset.ToTensor()
                ])):
    if train:
        face_dataset = dataset.FaceScoreDataset(limited=limited,
                                                image_list=image_list,
                                                transform = transform)

    else:
        face_dataset = dataset.FaceScoreDataset(limited=limited,
                                                image_list = image_list,
                                                train=False)
    return face_dataset

def get_live_dataset(live_train='',
                     live_test='',
                     transform=transforms.Compose([
                         dataset.RandomCrop(32),
                         dataset.ToTensor()
                     ])):
    return dataset.LiveDataset(live_train=live_train,
                               live_test=live_test,
                               transform=transform)


def get_dataloader(face_dataset, batch_size, shuffle=True, num_workers=4):

    return DataLoader(face_dataset, batch_size=batch_size,
                     shuffle=shuffle, num_workers=num_workers)


def np_load(path):
    print(path)
    return np.load(path)


def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)

    return (image - mean) / (std + 1e-6)


def evaluate_on_metric(hypo, score, log=True):
    '''
    
    :param hypo: numpy array
    :param score: numpy array
    :return: 
    '''

    if type(hypo) == list:
        hypo = np.array(hypo).reshape([-1])

    if type(hypo) == Tensor:
        hypo = hypo.detach().cpu().numpy().reshape([-1])
    if type(score) == Tensor:
        score = score.detach().cpu().numpy().reshape([-1])

    # debug: make sure data formats
    debug=0
    if debug:
        print(hypo.shape)
        print(score.shape)

    lcc = stats.pearsonr(hypo, score)[0]
    srocc = stats.spearmanr(hypo, score)[0]

    if str(lcc) == 'nan' or str(srocc) == 'nan':
        print(hypo, score)

    if log:
        log_print('LCC:{:.6f}'.format(lcc))
        log_print('SROCC:{:.6f}'.format(srocc))
    return lcc, srocc


def log_print(suffix=''):

    second = time.time() - start
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)

    str_time = '[{}:{}:{}]'.format(int(hour), int(minute), int(second))

    print(str_time + ' ' + suffix)


def save_model(model ,model_name='default', epoch=0):

    path = './model/'
    model_path = path + model_name + '_epoch_' + str(epoch)

    log_print('SAVING MODEL: ' + model_path)
    torch.save(model, model_path)


def exist_file(path, quit=False):
    if osp.isfile(path):
        return path
    elif quit:
        exit(0)