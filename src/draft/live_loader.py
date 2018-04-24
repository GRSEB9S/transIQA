import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.ion()

live_train = './data/live_generator/ft_live_train.txt'

train_live = [line.rstrip('\n').split()[0] for line in open(live_train)]
train_scores = [line.rstrip('\n').split()[1] for line in open(live_train)]

for i in train_live:

    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(img)

    plt.imshow(img)
    plt.show()
    plt.pause(0.1)
