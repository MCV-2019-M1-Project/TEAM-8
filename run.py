import ml_metrics as metrics
from dataset import Dataset
from dataset import HistDataset
from dataset import MaskDataset
import distance as dist
from utils import (
    calc_similarities,
    get_tops,
    get_groundtruth,
    normalize_hist,
    get_mask_metrics,
)
import matplotlib.pyplot as plt
import cv2
import numpy as np

"""For background removal vis HLS values go to dataset.py and check True, didn't have time to put it here cleanly"""

groundTruth = get_groundtruth("datasets/qsd2_w1/gt_corresps.pkl")
QS = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
QS2 = [
    normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd2_w1", masking=True)
]
DB = [
    normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB", masking=False)
]
gt_masks = MaskDataset("datasets/qsd2_w1")

k = 10

sims = calc_similarities(dist.canberra, DB, QS2, True)
tops = get_tops(sims, k)
mapAtK = metrics.mapk(groundTruth, tops, k)

print(str(tops[0]))
print(str(tops[1]))
print(str(tops[2]))

print("Map@k is " + str(mapAtK))

# TODO predicted_masks should be replaced by the array of masks that we compute
predicted_masks = gt_masks
mask_metrics = get_mask_metrics(predicted_masks, gt_masks)

print("Precision: " + str(mask_metrics["precision"]))
print("Recall: " + str(mask_metrics["recall"]))
print("F1-score: " + str(mask_metrics["f1_score"]))

# If you want to display any specific histogram
# R=DB[87][0]
# G=DB[87][1]
# B=DB[87][2]
# plt.plot(R,'r',G,'g',B,'b')
# plt.ylabel('Histogram')
# plt.show()

