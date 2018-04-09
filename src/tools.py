import matplotlib.pyplot as plt
from skimage import io
plt.figure()


def read_txt(image_path):

    images = [line.rstrip('\n').split()[0] for line in open(image_path)]
    scores = [line.rstrip('\n').split()[1] for line in open(image_path)]

    return images, scores

def show_image(image, title='default', commit=''):
    plt.title(str(title))
    plt.imshow(image)
    if commit != '':
        print(commit)
    plt.pause(0.01)
    plt.show()

def show_image_depth(image_path):
    image = io.imread(image_path)
    path = ''
    if len(image.shape)!=3:
        path = image_path
        print(path, image.shape)

    return path
