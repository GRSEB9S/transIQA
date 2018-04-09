import matplotlib.pyplot as plt
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