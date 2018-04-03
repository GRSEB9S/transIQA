# exam the images.txt file
# plot each picture with it's prediction score.
# for each picture, 0.5 second is given

import matplotlib.pyplot as plt
import numpy as np

image_list = './data/print_face.txt'
image_list = './data/print_live_filted_full.txt'

images = [line.rstrip('\n').split()[0] for line in open(image_list)]
scores = [line.rstrip('\n').split()[1] for line in open(image_list)]

for i in range(len(images)):
    #image = np.transpose(cv2.imread(images[i]), [2, 1, 0]) #cv2.imread() BGR form
    image = plt.imread(images[i])
    print(type(image))
    print(images[i])
    plt.imshow(image)
    plt.title(scores[i])
    plt.show()
    #plt.pause(0.001)
