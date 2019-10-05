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

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

i,j=np.where(thresh == 255)
y1=i[0]#fist point
x1=j[0]
y4=i[-1]#last point
x4=j[-1]
print(i)
print(j)
# for i in range(2000):
#     print(thresh[100,i])
k,d=np.where(thresh[:,0:100] == 255)
y2=k[0]#second point
x2=d[0]
y3=k[-1]#third point
x3=d[-1]

points=np.array([[11,  13],
                [14, 16],
                [17,  11],
                [12,  15]]).astype('int32')
print(points)
points[1]=(j[0],i[0])
points[2]=(d[0],k[0])
points[3]=(d[-1],k[-1])
points[0]=(j[-1],i[-1])
print(points)

image_countours=cv2.fillPoly(thresh, np.int32([points]) ,(255,255,255) , 8, 0, None)
image_countours=cv2.resize(image_countours,(1920,1080))
cv2.imshow("dd", image_countours)
cv2.waitKey(0)

# print("dd")
# print(k)
# plt.imshow(thresh)
# plt.show()
# cv2.waitKey(0)


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




