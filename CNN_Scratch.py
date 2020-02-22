import cv2 as cv
from matplotlib import pyplot as plt
import math
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# https://www.tensorflow.org/tutorials/images/cnn


# Homography matrix 
H = [[-1.02359791e+02, -3.46697336e+01,  1.04427550e+04],
    [-1.08731118e+01, -8.68669322e+01,  4.64350508e+03],
    [-2.26311453e-01, -2.44301016e-01,  1.00000000e+00]]


# Load in images
img1 = cv.imread('ori.png')
img2 = cv.imread('new.png')
print(img1.shape, img2.shape)


# Verity homography matrix
HInverse = np.linalg.inv(H)
img1Warped = cv.warpPerspective(img1, HInverse, (img1.shape[0], img1.shape[1]))
print((img2==img1Warped).sum() / (224*224*3))


# Model  

def euclidean_distance(y_true, y_pred):
    """L2 loss"""
    return math.sqrt(sum([(y_true - y_pred) ** 2 for a, b in zip(y_true, y_pred)]))


model_cnn = models.Sequential()

model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))

model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))

model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))


model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.Conv2D(128, (3, 3), activation='relu'))
model_cnn.add(layers.MaxPooling2D((2, 2)))


# First Fully connected layers       
model_cnn.add(layers.Flatten())
model_cnn.add(layers.Dense(1024, activation='relu'))
model_cnn.add(layers.Dropout(0.5))

# Last layer
model_cnn.add(layers.Dense(8, name='loss'))

print(model_cnn.summary())
