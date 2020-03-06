import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

print(tf.__version__)

""" Generate the dataset for rigid image registration (using MS-COCO dataset) """
# imgList = []
#
# numOfFile = 2000
# count = 0
# for path in glob.glob("./val2017/*.jpg"):
#     if count < numOfFile:
#         imgList.append(cv.imread(path))
#     count += 1
# print('image set length: ')
# print(len(imgList))
#
# # Random Projective transformation
#
# # Get the minimum height and width of all images in the image set.
# height_min = 100000
# for img in imgList:
#     if img.shape[0] < height_min:
#         height_min = img.shape[0]
#
# width_min = 100000
# for img in imgList:
#     if img.shape[1] < width_min:
#         width_min = img.shape[1]
#
# print(height_min, width_min);
#
# # Transform the color scheme from BGR to gray scale.
# for i in range(len(imgList)):
#     imgList[i] = cv.cvtColor(imgList[i], cv.COLOR_BGR2GRAY)
#
# # cv.imshow('image', imgList[200])
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# # Generate the training set.
# trainingSet = []
# badpair = 0
# temp = None
# for img in imgList:
#     bound = 20
#     sqrLength = 100
#     topLeftCornerX = random.randrange(bound + 1, img.shape[1] - bound - sqrLength)
#     topLeftCornerY = random.randrange(bound + 1, img.shape[0] - bound - sqrLength)
#     topRightCornerX = topLeftCornerX + sqrLength
#     topRightCornerY = topLeftCornerY
#     botLeftCornerX = topLeftCornerX
#     botLeftCornerY = topLeftCornerY + sqrLength
#     botRightCornerX = topLeftCornerX + sqrLength
#     botRightCornerY = topLeftCornerY + sqrLength
#     # print(botRightCornerY-topRightCornerY)
#     patchA = img[topLeftCornerY:botLeftCornerY, topLeftCornerX:topRightCornerX]
#     # cv.imshow('image', patchA)
#     # cv.waitKey(0)
#     # cv.destroyAllWindows()
#     topLeftRandomX = random.randrange(-bound, bound)
#     topLeftRandomY = random.randrange(-bound, bound)
#     topRightRandomX = random.randrange(-bound, bound)
#     topRightRandomY = random.randrange(-bound, bound)
#     botLeftRandomX = random.randrange(-bound, bound)
#     botLeftRandomY = random.randrange(-bound, bound)
#     botRightRandomX = random.randrange(-bound, bound)
#     botRightRandomY = random.randrange(-bound, bound)
#
#     newTopLeftCornerX = topLeftCornerX + topLeftRandomX
#     newTopLeftCornerY = topLeftCornerY + topLeftRandomY
#     newTopRightCornerX = topRightCornerX + topRightRandomX
#     newTopRightCornerY = topRightCornerY + topRightRandomY
#     newBotLeftCornerX = botLeftCornerX + botLeftRandomX
#     newBotLeftCornerY = botLeftCornerY + botLeftRandomY
#     newBotRightCornerX = botRightCornerX + botRightRandomX
#     newBotRightCornerY = botRightCornerY + botRightRandomY
#
#     src = np.float32([
#         [topLeftCornerX, topLeftCornerY],
#         [topRightCornerX, topRightCornerY],
#         [botLeftCornerX, botLeftCornerY],
#         [botRightCornerX, botRightCornerY]
#     ])
#
#     dst = np.float32([
#         [newTopLeftCornerX, newTopLeftCornerY],
#         [newTopRightCornerX, newTopRightCornerY],
#         [newBotLeftCornerX, newBotLeftCornerY],
#         [newBotRightCornerX, newBotRightCornerY]
#     ])
#     H = cv.getPerspectiveTransform(src, dst)
#
#     HInverse = np.linalg.inv(H)
#
#     imgWarped = cv.warpPerspective(img, HInverse, (img.shape[0], img.shape[1]))
#
#     patchB = imgWarped[topLeftCornerY:botLeftCornerY, topLeftCornerX:topRightCornerX]
#
#     if patchA.shape[0] != patchB.shape[0] or patchA.shape[1] != patchB.shape[1]:
#         badpair += 1
#     else:
#         patchAReshaped = np.reshape(patchA, -1)
#         patchBReshaped = np.reshape(patchB, -1)
#         label = np.float32([
#             [topLeftRandomX, topLeftRandomY],
#             [topRightRandomX, topRightRandomY],
#             [botLeftRandomX, botLeftRandomY],
#             [botRightRandomX, botRightRandomY]
#         ])
#         labelReshaped = np.reshape(label, -1)
#         dataReshaped = np.append(patchAReshaped, patchBReshaped)
#         dataReshaped = np.append(dataReshaped, labelReshaped)
#         trainingSet.append(dataReshaped)
#
# print(badpair)
# print(trainingSet[0].shape)
# # print(trainingSet[100][0].shape)
# # print(trainingSet[100][1].shape)
# # print(trainingSet[100][2].shape)
# # cv.imshow('image', trainingSet[100][0])
# # cv.waitKey(0)
# # cv.destroyAllWindows()
# # cv.imshow('image', trainingSet[100][1])
# # cv.waitKey(0)
# # cv.destroyAllWindows()
#
# f = open("train.txt", "w")
# for imgPair in trainingSet:
#     for i in range(imgPair.shape[0]):
#         f.write(str(imgPair[i]) + " ")
#     f.write('\n')
# f.close()


"""======== Using basic TensorFlow to do the regression model ==========="""

""" Read data from txt file"""
data = np.loadtxt("./train.txt")
print(data.shape)

imgs = data[:, :-8]
print(imgs.shape)
labels = data[:, -8:]
print(labels.shape)

train_imgs = imgs
train_labels = labels
# val_imgs = imgs[1000:]
# val_labels = labels[1000:]

print(train_imgs.shape)
print(train_labels.shape)
# print(val_imgs.shape)
# print(val_labels.shape)

train_imgs = train_imgs / 255.0
# val_imgs = val_imgs / 255.0

def build_model():
    model = keras.Sequential([
        layers.Dense(1000, activation='relu', input_shape=(20000,)),
        layers.Dense(500, activation='relu'),
        layers.Dense(8)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model

    # regressor = tf.estimator.BaselineRegressor()
    # return regressor


model = build_model()
print(model.summary())


# def input_fn_train():
#     return (tf.constant(train_imgs), tf.constant(train_labels))
#
# def input_fn_val():
#     return (tf.constant(val_imgs), tf.constant(val_labels))
#
#
# model.train(input_fn=input_fn_train)
#
# loss = model.evaluate(input_fn=input_fn_val)["loss"]

# example_batch = train_imgs[0:10]
# example_result = model.predict(example_batch)
# print(example_result)
# example_true = train_labels[0:10]
# print(example_true)

# print(loss)

EPOCHS = 100

history = model.fit(
  train_imgs, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)

print(history.history)

plt.plot(history.history['loss'])
plt.legend(["training_loss", "validation_loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss plot")
plt.show()






# imgs = tf.data.Dataset.from_tensor_slices(data[:, :-9])
# labels = tf.data.Dataset.from_tensor_slices(data[:, -9:])
# print(len(list(imgs)))
# print(imgs.element_spec)
# print(labels.element_spec)

# train_imgs = imgs.take(1000)
# train_labels = labels.take(1000)
# val_imgs = imgs.skip(1000)
# val_labels = labels.skip(1000)

# print(len(list(train_imgs)))
# print(len(list(val_labels)))
# print(train_imgs.element_spec)
# print(val_labels.element_spec)

# max = 0
# for elem in imgs:
#     for value in elem.numpy():
#         if value > max:
#             max = value
# print(max)

# Normalization
# train_imgs = train_imgs / 255.0
# val_imgs = val_imgs / 255.0

# def build_model():
#     model = keras.Sequential([
#         layers.Dense(1000, activation='relu', input_shape=(20000,)),
#         layers.Dense(500, activation='relu'),
#         layers.Dense(9)
#     ])
#
#     optimizer = tf.keras.optimizers.RMSprop(0.001)
#
#     model.compile(loss='mse',
#                   optimizer=optimizer,
#                   metrics=['mae', 'mse'])
#     return model
#
# model = build_model()
#
# print(model.summary())
#
# example_batch = train_imgs.take(10)
# example_result = model.predict(example_batch)
# print(example_result)










""" The following part are learning part. """
# img1 = cv.imread('IMG_1.jpg')
# img2 = cv.imread('IMG_2.jpg')
# img_cpy = cv.imread('img_cpy.jpg')

# cv.imwrite('img_cpy.jpg', img)
# print('image 1 shape: ')
# print(img1.shape)
# print('image 2 shape: ')
# print(img2.shape)

# print('image size: ')
# print(img.size)
# print('image data type: ')
# print(img.dtype)

"""
Image slice
"""
# img2Slice = img2[50:600, 300:800]
# cv.imshow('image', imgSlice)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Image channel extraction
"""
# r, g, b = cv.split(img)
# img_cpy[:, :, 1] = 0
# img_cpy[:, :, 2] = 0

"""
Align two images with transparency (two images must have the same shape)
"""
# blend = cv.addWeighted(img, 0.7, img_cpy, 0.3, 0)
#
# cv.imshow('image', blend)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
Cover an image with another one without transparency.
"""
# # I want to put logo on top-left corner, So I create a ROI
# img2Slice = img2[50:600, 300:800]
# rows, cols, channels = img2Slice.shape
# roi = img1[0:rows, 0:cols]
#
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv.cvtColor(img2Slice, cv.COLOR_BGR2GRAY)
# ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# mask_inv = cv.bitwise_not(mask)
#
# # Now black-out the area of logo in ROI
# img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
#
# # Take only region of logo from logo image.
# img2_fg = cv.bitwise_and(img2Slice, img2Slice, mask=mask)
#
# # Put logo in ROI and modify the main image
# dst = cv.add(img1_bg, img2_fg)
# img1[0:rows, 0:cols] = dst
#
# cv.imshow('image', img1)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
Transformations
"""
# Scaling
# img1Shrink = cv.resize(img1, None, fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
# cv.imshow('image', img1Shrink)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Translation
# rows1, cols1 = img1.shape[:2]
# # [1, 0, tx] [0, 1, ty]
# M = np.float32([[1, 0, 100],[0, 1, 50]])
# img1Translation = cv.warpAffine(img1, M, (cols1, rows1))
# cv.imshow('image', img1Translation)
# cv.waitKey(0)
# cv.destroyAllWindows()

# Rotation
# rows1, cols1 = img1.shape[:2]
# # getRotationMatrix2D((rotCenterX, rotCenterY), degree, scale)
# M = cv.getRotationMatrix2D(((cols1-1)/2.0, (rows1-1)/2.0), 90, 1)
# img1Rotation = cv.warpAffine(img1, M, (cols1, rows1))
# cv.imshow('image', img1Rotation)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Affine Transformation
"""
# rows1, cols1 = img1.shape[:2]
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
# M = cv.getAffineTransform(pts1, pts2)
# dst = cv.warpAffine(img1, M, (cols1, rows1))
# cv.imshow('image', dst)
# cv.waitKey(0)
# cv.destroyAllWindows()

"""
Corner Detection
"""
# Harris Corner Detection
# img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# img1Gray = np.float32(img1Gray)
# dst = cv.cornerHarris(img1Gray, 2, 3, 0.04)
# # result is dilated for marking the corners, not important
# dst = cv.dilate(dst, None)
# # Threshold for an optimal value, it may vary depending on the image.
# img1[dst > 0.01 * dst.max()] = [0, 0, 255]
# img1Shrink = cv.resize(img1, None, fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
# cv.imshow('image', img1Shrink)
# cv.waitKey(0)
# cv.destroyAllWindows()


# Shi-Tomasi Corner Detection
# img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# corners = cv.goodFeaturesToTrack(img1Gray, 200, 0.01, 10)
# corners = np.int0(corners)
# for i in corners:
#     x, y = i.ravel()
#     cv.circle(img1, (x, y), 3, (0, 0, 255), -1)
#
# img1Shrink = cv.resize(img1, None, fx=0.5, fy=0.5, interpolation = cv.INTER_CUBIC)
# cv.imshow('image', img1Shrink)
# cv.waitKey(0)
# cv.destroyAllWindows()


"""
Using matplotlib to show images.
"""
# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

"""
Create a black image
"""
# img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
# cv.line(img,(0,0),(511,511),(0,0,255),5)
#
# plt.imshow(img)
# plt.show()
