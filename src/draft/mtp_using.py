import multiprocessing as mtp
import numpy as np


def np_load(input):

    return np.load(input)

def np_load():



p = mtp.Pool(4)
img = p.map(np_load, ('../dataset/transIQA/faces/aflw__face_48129_pristine_images_1.npy',
                      '../dataset/transIQA/faces/aflw__face_48129_GB1_1.npy'))

print(img[0].shape)
print(img[1].shape)