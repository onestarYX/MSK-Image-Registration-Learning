import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
import glob


img1 = cv.imread('IMG_1.jpg')
img2 = cv.imread('IMG_2.jpg')
img_cpy = cv.imread('img_cpy.jpg')

imgList = []

for path in glob.glob("./val2017/*.jpg"):
    imgList.append(cv.imread(path))
# cv.imwrite('img_cpy.jpg', img)
print('image 1 shape: ')
print(img1.shape)
print('image 2 shape: ')
print(img2.shape)
print('image set length: ')
print(len(imgList))
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


# Random Projective transformation
height_min = 100000
for img in imgList:
    if img.shape[0] < height_min:
        height_min = img.shape[0]

width_min = 100000
for img in imgList:
    if img.shape[1] < width_min:
        width_min = img.shape[1]

print(height_min, width_min);

img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
bound = 20
sqrLength = 80
topLeftCornerX = random.randrange(bound + 1, img1.shape[1] - bound - sqrLength)
topLeftCornerY = random.randrange(bound + 1, img1.shape[0] - bound - sqrLength)
topRightCornerX = topLeftCornerX + sqrLength
topRightCornerY = topLeftCornerY
botLeftCornerX = topLeftCornerX
botLeftCornerY = topLeftCornerY + sqrLength
botRightCornerX = topLeftCornerX + sqrLength
botRightCornerY = topLeftCornerY + sqrLength
# print(botRightCornerY-topRightCornerY)
patchA = img1Gray[topLeftCornerY:botLeftCornerY, topLeftCornerX:topRightCornerX]
cv.imshow('image', patchA)
cv.waitKey(0)
cv.destroyAllWindows()
topLeftRandomX = random.randrange(-bound, bound)
topLeftRandomY = random.randrange(-bound, bound)
topRightRandomX = random.randrange(-bound, bound)
topRightRandomY = random.randrange(-bound, bound)
botLeftRandomX = random.randrange(-bound, bound)
botLeftRandomY = random.randrange(-bound, bound)
botRightRandomX = random.randrange(-bound, bound)
botRightRandomY = random.randrange(-bound, bound)

newTopLeftCornerX = topLeftCornerX + topLeftRandomX
newTopLeftCornerY = topLeftCornerY + topLeftRandomY
newTopRightCornerX = topRightCornerX + topRightRandomX
newTopRightCornerY = topRightCornerY + topRightRandomY
newBotLeftCornerX = botLeftCornerX + botLeftRandomX
newBotLeftCornerY = botLeftCornerY + botLeftRandomY
newBotRightCornerX = botRightCornerX + botRightRandomX
newBotRightCornerY = botRightCornerY + botRightRandomY

src = np.float32([
    [topLeftCornerX, topLeftCornerY],
    [topRightCornerX, topRightCornerY],
    [botLeftCornerX, botLeftCornerY],
    [botRightCornerX, botRightCornerY]
])

dst = np.float32([
    [newTopLeftCornerX, newTopLeftCornerY],
    [newTopRightCornerX, newTopRightCornerY],
    [newBotLeftCornerX, newBotLeftCornerY],
    [newBotRightCornerX, newBotRightCornerY]
])
H = cv.getPerspectiveTransform(src, dst)
print(H)

HInverse = np.linalg.inv(H)
# img1Warped = cv.warpPerspective(img1, HInverse, (600, 600))
# cv.imshow('image', img1Warped)
# cv.waitKey(0)
# cv.destroyAllWindows()

img1Warped = cv.warpPerspective(img1Gray, HInverse, (img1Gray.shape[0], img1Gray.shape[1]))
# cv.imshow('image', img1Warped)
# cv.waitKey(0)
# cv.destroyAllWindows()

patchB = img1Warped[topLeftCornerY:botLeftCornerY, topLeftCornerX:topRightCornerX]
cv.imshow('image', patchB)
cv.waitKey(0)
cv.destroyAllWindows()











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
