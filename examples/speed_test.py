# test speed with lower api

import face_iqa
import time
import glob
import os
import sys
import cv2

input_dir = '/home/junrui/github/dataset/LS3D-W/AFLW2000-3D-Reannotated/'

model_path = '/data/junrui/github/transIQA/model/ft12/tid2013_mse_423_0.8354_0.8103.pth.tar'
fi = face_iqa.FaceIQA(enable_cuda=True, model_path=model_path)

types = ('*.jpg', '*.png')
images_list = []
for files in types:
    images_list.extend(glob.glob(os.path.join(input_dir, files)))
images = []
for i in images_list:
    images.append(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB))

print('Starting processing')
start = time.time()
num_images = len(images)
for i in range(1, num_images+1):

    fi.get_score(images[i-1])

    percentage = 100 * i / num_images
    item = '[====={:.1f}%=====]({}/{}) {:.1f}fps'.format(
        percentage,
        i,
        num_images,
        i / (time.time() - start)
    )
    print("\033[K" + item + "\r", end="")
    sys.stdout.flush()
print('')