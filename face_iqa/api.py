import torch
import os
import cv2
import numpy as np
from .model import *
from .utils import *


class FaceIQA:
    """Initialize the face iqa pipline

    Args:

    """

    def __init__(self, enable_cuda=True,
                 model_path='/data/junrui/github/transIQA/model/ft12/live_mse_test_20_0.6461_0.6751.pth.tar',
                 ):
        self.enable_cuda = enable_cuda
        self.model_path = model_path
        self.dtype = torch.float

        if self.enable_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize the iqa model
        if not os.path.isfile(self.model_path):
            print('[ERROR]No model on path: {}'.format(
                self.model_path
            ))
            exit(0)

        self.face_iqa_net = ft12(Net_deep())
        checkpoint = torch.load(model_path)
        self.face_iqa_net.load_state_dict(checkpoint['state_dict'])
        self.face_iqa_net.to(self.device)
        self.face_iqa_net.eval()

    def get_score(self, input_image):
        """Run the model prediction with one image(path or np array)

        :param input_image: path or image
        :return: mean score of an image
        """
        if isinstance(input_image, str):
            try:
                image = cv2.imread(input_image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except IOError:
                print("error opening file :: ", input_image)
                return None
        else:
            image = input_image

        # prepare image
        image = standardize_image(image)

        height = image.shape[0]
        width = image.shape[1]

        patches = []
        for i in range(30):  # num of small patch
            top = np.random.randint(0, height - 32)
            left = np.random.randint(0, width - 32)
            patches.append(image[top:top + 32, left:left + 32, :].transpose((2, 0, 1)))

        patches = np.array(patches)
        patches = torch.from_numpy(patches).to(self.device, self.dtype)

        scores = self.face_iqa_net(patches).detach()
        score = sum(scores) / 30

        return score

    def process_list(self, input_list):
        """Process list of input(txt or np arrays)

        :param input_list: txt or np arrays
        :return: scores
        """

        assert isinstance(input_list, list)

        scores = []

        for input_image in input_list:
            scores.append(self.predict_score(input_image))

        return scores
