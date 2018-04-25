from torch.utils.data import Dataset
import numpy as np
import torch
import tools
import cv2
import multiprocessing as mtp

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


class FaceScoreDataset(Dataset):
    """Face Score dataset"""

    def __init__(self, image_list, transform=None, limited=True, train=True):
        """
        initiate imagelist and score
        :param image_list: .txt file with image path and score
        """
        faces = [line.rstrip('\n').split()[0] for line in open(image_list)]
        scores = [line.rstrip('\n').split()[1] for line in open(image_list)]
        self.limited = limited
        self.transform = transform
        self.images = []
        self.scores = []
        self.train = train

        if self.limited:
            num_faces = 2000
        else:
            num_faces = 10000

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

        # reading datasets
        assert num_faces < len(faces)
        assert len(faces) > 50000
        if self.train:
            tools.log_print('Loading Training set')
            for i in np.random.choice(50000, num_faces):
                #debug
                debug=0
                if debug:
                    print(i)
                self.images.append(tools.standardize_image(
                    np.array(np.load(faces[i]), dtype=np.float32)))
                self.scores.append(scores[i])
        else:
            tools.log_print('Loading Testing set')
            test_length = 8000
            if self.limited:
                test_length = 4000

            num = 0
            self.scores = np.zeros(test_length)
            for i in range(len(faces) - test_length, len(faces)):
                self.images.append(tools.standardize_image(
                    np.array(np.load(faces[i]), dtype=np.float32)))

                self.scores[i - len(faces) + test_length] = float(scores[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        assert self.train == True, 'getitem fuction only for training.'

        image = self.images[idx]

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


class LiveDataset(Dataset):
    """LIVE dataset"""

    def __init__(self, live_train, live_test, transform=None, limited=True, train=True):
        """
        initiate image list and score
        """
        train_live = [line.rstrip('\n').split()[0] for line in open(live_train)]
        train_scores = [line.rstrip('\n').split()[1] for line in open(live_train)]
        test_live = [line.rstrip('\n').split()[0] for line in open(live_test)]
        test_scores = [line.rstrip('\n').split()[1] for line in open(live_test)]
        self.limited = limited
        self.transform = transform
        self.train_images = []
        self.train_scores = []
        self.test_images = []
        self.test_scores = []

        # debug: show image shape
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

        # reading datasets
        tools.log_print('Loading Training set')
        num_train = len(train_live)
        for i in np.random.choice(num_train, num_train):
            #debug
            debug=0
            if debug:
                print(i)
            img = cv2.imread(train_live[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.train_images.append(tools.standardize_image(
                np.array(img, dtype=np.float32)))
            self.train_scores.append(train_scores[i])

        tools.log_print('Loading Testing set')
        num_test = len(test_live)
        self.test_scores = np.zeros(num_test)
        for i in range(num_test):
            img = cv2.imread(test_live[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.test_images.append(tools.standardize_image(
                np.array(img, dtype=np.float32)))
            self.test_scores[i] = float(test_scores[i])

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):

        image = self.train_images[idx]

        # debug
        debug = 0
        if debug:
            print(image.dtype)
            print(np.array(image, dtype=np.float32).dtype)
            print(image.shape)
            tools.show_image(np.array(image, dtype=np.int)) # interfaces is not supported for multi-processing
            #exit(0)

        score = np.array((float(self.train_scores[idx])), dtype=np.float32).reshape([1])#IMPORTANT
        sample = {'image': image, 'score': score}

        if self.transform:
            sample = self.transform(sample)

        return sample