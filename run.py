import ml_metrics as metrics
from dataset import Dataset
from dataset import HistDataset
import distance as dist
import cv2
import numpy as np
from utils import calc_similarities, get_tops, get_groundtruth, normalize_hist,edgedetector,background_removal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
'''For background removal vis HLS values go to dataset.py and check True, didn't have time to put it here cleanly'''

groundTruth = get_groundtruth("datasets/qsd1_w1/gt_corresps.pkl")

# QS = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
QS2 = Dataset("datasets/qsd2_w1")
# DB = [normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB")]

k = 10

# sims = calc_similarities(dist.canberra, DB, QS, True)
# tops = get_tops(sims, k)
# mapAtK = metrics.mapk(groundTruth, tops, k)
#
# print(str(tops[0]))
# print(str(tops[1]))
# print(str(tops[2]))
#
# print("Map@k is " + str(mapAtK))



img=cv2.GaussianBlur(QS2[1], (5,5),0)

hist = cv2.calcHist([img],[0],None,[256],[0,256])
i, j=np.where(hist == max(hist))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,i,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


# image_countours=cv2.fillPoly(thresh, countours,(255,255,255) , 8, 0, None)
# cv2.imshow("dd", thresh)
plt.imshow(thresh)
plt.show()
print(np.where(thresh[0][:] == 255))


cv2.waitKey(0)


# print(QS2[1][:,:,1])
# imgplot=plt.imshow(QS2[1])
# cv2.imshow('red',QS2[1][:,:,0])
# cv2.imshow('green',QS2[1][:,:,1])
# cv2.imshow('blue',QS2[1][:,:,2])
# imgray = cv2.cvtColor(QS2[1], cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 100, 255, 0)
# countours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# image_countours=cv2.drawContours(QS2[1],countours,-1, (0,255,0), 3)
# image_countours=cv2.resize(image_countours,(1920,1080))
# cv2.imshow('countures',image_countours)
# print(countours)
# imgplot=plt.imshow(cv2.cvtColor(QS2[1], cv2.COLOR_BGR2RGB))
# plt.show()
# cv2.waitKey(0)



# edgeImg = np.max(np.array([edgedetector(denoised_dataset[1][:, :, 0]), edgedetector(denoised_dataset[1][:, :, 1]), edgedetector(denoised_dataset[1][:, :, 2])]),axis=0)
# cv2.imshow('countures', edgeImg)




