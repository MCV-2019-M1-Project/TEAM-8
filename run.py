import ml_metrics as metrics
import cv2
from dataset import HistDataset
from dataset import MaskDataset
from matplotlib import pyplot as plt
import distance as dist
import text_removal
#from utils import (
#    calc_similarities,
#    get_tops,
#    get_groundtruth,
#    normalize_hist,
#    get_mask_metrics,
#)
# namedataset=[text_removal.getpoints2(im) for im in text_removal.text_remover("datasets/qsd1_w2")]
#
# #
# # ##Fer el threshold per colors, ja que el valor de igual nomes el tenen el retols.
# image_countours=cv2.resize(namedataset[1],(1280,
#                                           700))
# cv2.imshow('try', image_countours)
# cv2.waitKey(0)
# #

namedataset=[text_removal.getpoints2(im) for im in text_removal.text_remover("datasets/qsd1_w2")]
print(namedataset)
# print(len(namedataset))
# for im in range(len(namedataset)):
#     print(im)
#     cv2.imwrite("outputs/"+str(im)+".png",namedataset[im])

# histogramas=[text_removal.getpoints4(im)for im in text_removal.text_remover("datasets/qsd1_w2")]
# print(type(histogramas[0][0]))
# plt.plot(histogramas[0])
# plt.show()
# cv2.imshow('try', histogramas[0])
# cv2.waitKey(0)
# print(histogramas[1])
##Fer el threshold per colors, ja que el valor de igual nomes el tenen el retols.
#image_countours=cv2.resize(namedataset[1],(1280,
                                           #700))







# dataset1=[text_removal.getimg(im) for im in text_removal.text_remover("datasets/qsd1_w2")]

#hsv = cv2.cvtColor(dataset1[1], cv2.COLOR_BGR2HSV)
# hist=cv2.calcHist(dataset1[0], [0], None, [256], [0, 256])
# image_countours=cv2.resize(hsv,(1280, 700))
#cv2.imshow('try', image_countours)
# cv2.waitKey(0)
# print(type(dataset1[0][:,:,0]))
# cv2.imshow('try', dataset1[0][:,:,0])
# cv2.waitKey(0)
# plt.plot(dataset1[0][:,:,0])
# plt.show()







