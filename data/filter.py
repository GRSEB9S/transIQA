# image size[1] | size[2] > 400
# image file size > 20k

import os.path as osp
import cv2

size_gate = 20000
image_list = './data/images.txt'

images = [line.rstrip('\n').split()[0] for line in open(image_list)]
scores = [line.rstrip('\n').split()[1] for line in open(image_list)]

high_images = []

for i in range(len(images)):
    size = osp.getsize(images[i])
    if size < size_gate:
        continue
    shape = cv2.imread(images[i]).shape
    if shape[0] < 400 or shape[1] < 400:
        continue
    high_images.append(images[i] + ' ' + scores[i])
    print(len(high_images))

with open('./data/high_images.txt', 'w') as f:
    for i in high_images:
        f.write(i + '\n')
