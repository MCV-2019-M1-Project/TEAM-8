import ml_metrics as metrics
from dataset import Dataset
from dataset import HistDataset
import distance as dist
from utils import calc_similarities, get_tops, get_groundtruth, normalize_hist
import matplotlib.pyplot as plt
import cv2
import numpy as np
'''For background removal vis HLS values go to dataset.py and check True, didn't have time to put it here cleanly'''

groundTruth = get_groundtruth("datasets/qsd2_w1/gt_corresps.pkl")
#QS = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
QS2 = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd2_w1", True)]
DB = [normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB", False)]

k = 10

sims = calc_similarities(dist.canberra, DB, QS2, True)
tops = get_tops(sims, k)
mapAtK = metrics.mapk(groundTruth, tops, k)

print(str(tops[0]))
print(str(tops[1]))
print(str(tops[2]))

print("Map@k is " + str(mapAtK))

#If you want to display any specific histogram
# R=DB[87][0]
# G=DB[87][1]
# B=DB[87][2]
# plt.plot(R,'r',G,'g',B,'b')
# plt.ylabel('Histogram')
# plt.show()

#for imgi in QS2:

"""
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    
    resized1 = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    resized2 = cv2.resize(image_countours, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("dd", resized1)
    cv2.imshow("dad", resized2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

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




