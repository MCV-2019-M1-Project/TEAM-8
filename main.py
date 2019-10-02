import numpy
import cv2
import os, os.path
from matplotlib import pyplot as plt
import glob

images = []
hist = []
thist = []

# n=0
for i in glob.glob("DDBB/Debug/*jpg"):
    img = cv2.imread(i)
    images.append(img)
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    histg = cv2.calcHist([img], [1], None, [256], [0, 256])
    histb = cv2.calcHist([img], [2], None, [256], [0, 256])

    thist.append(histr)
    thist.append(histg)
    thist.append(histb)
    hist.append(thist)

print(hist[0][1] == histg)

# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# img = cv2.imread("DDBB/bbdd_00000.jpg",cv2.COLOR_BGR2GRAY)
# hist = cv2.calcHist([img],[0],None,[256],[0,256])
# plt.plot(histg)
# plt.show()