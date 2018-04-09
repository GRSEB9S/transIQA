from torch.utils.data import Dataset
from skimage import io, transform
from torchvision import transforms
import numpy as np
import torch

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
        print(image.shape)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'score': torch.from_numpy(score)}


class FaceScoreDataset(Dataset):
    """Face Score dataset"""

    def __init__(self, image_list, transform=None):
        """
        initiate imagelist and score
        :param image_list: .txt file with image path and score
        """
        self.images = [line.rstrip('\n').split()[0] for line in open(image_list)]
        self.scores = [line.rstrip('\n').split()[1] for line in open(image_list)]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = io.imread(self.images[idx])
        score = np.array((float(self.scores[idx]))).reshape([1])#IMPORTANT
        sample = {'image': image, 'score': score}

        if self.transform:
            sample = self.transform(sample)

        return sample