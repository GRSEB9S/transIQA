
def read_txt(image_path):

    images = [line.rstrip('\n').split()[0] for line in open(image_path)]
    scores = [line.rstrip('\n').split()[1] for line in open(image_path)]

    return images, scores
