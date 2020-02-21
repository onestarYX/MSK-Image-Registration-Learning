from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.layers import Dense, Dropout, Activation
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import cv2 as cv
import numpy as np
import random
from matplotlib import pyplot as plt


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


################################################################################

# Play with pretrained VGG network 

model = VGG16()
print(model.summary())
plot_model(model, to_file='vgg.png')


# Reshape because we only load in one image
img_reshape = img1.reshape((1, img1.shape[0], img1.shape[1], img1.shape[2]))

# Prediction 
yhat = model.predict(img_reshape)


from keras.applications.vgg16 import decode_predictions # Not understood thus far 
# convert the probabilities to class labels
label = decode_predictions(yhat)
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))


################################################################################

# Homography Estimation 

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
model_cnn.add(layers.Dense(1024, activation='relu'))
model_cnn.add(layers.Dropout(0.5))


# Last layer (Regression layer: produce 8 real-valued numbers and L2 loss )


