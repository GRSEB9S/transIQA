import numpy as np


def standardize_image(image):
    mean = np.mean(image)
    std = np.std(image)

    return (image - mean) / (std + 1e-6)