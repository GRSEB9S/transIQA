from torch.utils.data import Dataset
from skimage import io, transform
from torchvision import transforms
import numpy as np
import torch
import tools
import glymur
import cv2
import dlib
import matplotlib.pyplot as plt

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, score = sample['image'], sample['score']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'score': score}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, score = sample['image'], sample['score']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'score': torch.from_numpy(score)}


class FaceScoreDataset_015(Dataset):
    """Face Score dataset"""

    def __init__(self, image_list, transform=None):
        """
        initiate imagelist and score
        :param image_list: .txt file with image path and score
        """
        self.images = [line.rstrip('\n').split()[0] for line in open(image_list)]
        self.scores = [line.rstrip('\n').split()[1] for line in open(image_list)]
        self.transform = transform

        #debug: show image shape
        debug = False
        if debug:
            num = 0
            path = []
            for i in self.images:
                fault_path = tools.show_image_depth(i)
                if fault_path != '':
                    path.append(fault_path)
                    num += 1
            print(num)
            print(path)
            exit(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        # Version 0.15
        if self.images[idx][-4:] == '.jp2':
            image = glymur.Jp2k(self.images[idx])[:]

            #debug
            debug=0
            if debug:
                tools.show_image(image)

        else:
            image = io.imread(self.images[idx])
        image = np.array(image, dtype=np.float32)
        """

        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)

        # debug
        debug = 0
        if debug:
            print(image.dtype)
            print(np.array(image, dtype=np.float32).dtype)
            print(image.shape)
            tools.show_image(np.array(image, dtype=np.int)) # interfaces is not supported for multi-processing
            #exit(0)

        score = np.array((float(self.scores[idx])), dtype=np.float32).reshape([1])#IMPORTANT
        sample = {'image': image, 'score': score}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FaceScoreDataset(Dataset):
    """Face Score dataset"""

    def __init__(self, image_list, transform=None, use_dlib_cnn=True, scale = 1.2):
        """
        initiate imagelist and score
        :param image_list: .txt file with image path and score
        """
        # CNNs has to be used.
        # not only frontal face is needed.
        #

        dlib_model_path = './model/mmod_human_face_detector.dat'

        self.images = [line.rstrip('\n').split()[0] for line in open(image_list)]
        self.scores = [line.rstrip('\n').split()[1] for line in open(image_list)]
        self.transform = transform
        self.face_detector = dlib.cnn_face_detection_model_v1(dlib_model_path)
        self.faces = []
        self.face_scores = []


        print('Datasets reading and Face detecting')
        # prepare faces
        # detect faces and save as images file

        # read images and detect faces
        pristine_images = []
        num_pristine = int(len(self.images) / 21)
        for i in range(num_pristine):
            pristine_images.append(self.images[i * 21])

        # detect faces for one pristine
        for i in range(len(pristine_images)):
            print('Reading images: %s'%pristine_images[i])
            image = cv2.imread(pristine_images[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.detect_faces(image)

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
                    image = cv2.imread(self.images[i*21 + j])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    score = self.scores[i*21 + j]
                    for location in face_locations:
                        #debug
                        debug=0
                        if debug:
                            plt.imshow(image[location[0]:location[1], location[2]:location[3], :])
                            plt.title(str(score))
                            plt.show()
                        self.faces.append(image[location[0]:location[1], location[2]:location[3], :])
                        self.face_scores.append(score)

        #debug: show image shape
        debug = False
        if debug:
            num = 0
            path = []
            for i in self.images:
                fault_path = tools.show_image_depth(i)
                if fault_path != '':
                    path.append(fault_path)
                    num += 1
            print(num)
            print(path)
            exit(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        # Version 0.15
        if self.images[idx][-4:] == '.jp2':
            image = glymur.Jp2k(self.images[idx])[:]

            #debug
            debug=0
            if debug:
                tools.show_image(image)

        else:
            image = io.imread(self.images[idx])
        image = np.array(image, dtype=np.float32)
        """

        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)

        # debug
        debug = 0
        if debug:
            print(image.dtype)
            print(np.array(image, dtype=np.float32).dtype)
            print(image.shape)
            tools.show_image(np.array(image, dtype=np.int)) # interfaces is not supported for multi-processing
            #exit(0)

        score = np.array((float(self.scores[idx])), dtype=np.float32).reshape([1])#IMPORTANT
        sample = {'image': image, 'score': score}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def detect_faces(self, image):

        return self.face_detector(image, 1)

