import ml_metrics as metrics
from dataset import Dataset
from dataset import HistDataset
from dataset import MaskDataset
import distance as dist
from utils import calc_similarities, get_tops, get_groundtruth, normalize_hist, get_mask_metrics
import matplotlib.pyplot as plt
import cv2
import numpy as np
'''For background removal vis HLS values go to dataset.py and check True, didn't have time to put it here cleanly'''

groundTruth = get_groundtruth("datasets/qsd2_w1/gt_corresps.pkl")
#QS = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
QS2 = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd2_w1", True)]
DB = [normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB", False)]
gt_masks = MaskDataset("datasets/qsd2_w1")

k = 10

sims = calc_similarities(dist.canberra, DB, QS2, True)
tops = get_tops(sims, k)
mapAtK = metrics.mapk(groundTruth, tops, k)

print(str(tops[0]))
print(str(tops[1]))
print(str(tops[2]))

print("Map@k is " + str(mapAtK))

#TODO predicted_masks should be replaced by the array of masks that we compute
predicted_masks = gt_masks
mask_metrics = get_mask_metrics(predicted_masks, gt_masks)

print("Precision: " + str(mask_metrics["precision"]))
print("Recall: " + str(mask_metrics["recall"]))
print("F1-score: " + str(mask_metrics["f1_score"]))

#If you want to display any specific histogram
# R=DB[87][0]
# G=DB[87][1]
# B=DB[87][2]
# plt.plot(R,'r',G,'g',B,'b')
# plt.ylabel('Histogram')
# plt.show()


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

