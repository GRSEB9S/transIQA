import matplotlib.pyplot as plt
from skimage import io
import numpy as np

file_list = [line.rstrip('\n') for line in open('./data/face_score_generated_dlib.txt')]
print(file_list[0])
print(len(file_list))
for line in file_list:
    im = np.load(line.split()[0])
    print(line.split()[0])
    plt.imshow(im)
    plt.title(line.split()[1])
    plt.show()

