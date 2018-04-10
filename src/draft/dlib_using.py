import dlib
import cv2
import torch
import matplotlib.pyplot as plt
import numpy

# Dlib


image_path = '../dataset/transIQA/pristine_images/aflw__face_43318.jpg'
model_path = './model/mmod_human_face_detector.dat'
cuda = 1

# read images
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

# dlib configuration
if cuda and torch.cuda.is_available():
    face_detector = dlib.cnn_face_detection_model_v1(model_path)
else:
    face_detector = dlib.get_frontal_face_detector()

dets = face_detector(image, 1)

faces = []

print("Number of faces detected: {}".format(len(dets)))
for i, d in enumerate(dets):
    if cuda and torch.cuda.is_available():
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
            i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        faces.append(image[d.rect.top():d.rect.bottom(), d.rect.left():d.rect.right(), :])
    else:
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        faces.append(image[d.top():d.bottom(), d.left():d.right(), :])

for i in faces:
    plt.imshow(i)
    plt.show()
