import ml_metrics as metrics
import cv2
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

"""For background removal vis HLS values go to dataset.py and check True, didn't have time to put it here cleanly"""

groundTruth = get_groundtruth("datasets/qsd2_w1/gt_corresps.pkl")
QS = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
QS2 = [
    normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd2_w1", masking=True)
]
DB = [
    normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB", masking=False)
]

mask_dataset = HistDataset("datasets/qsd2_w1", masking=True)

predicted_masks = [cv2.threshold(mask_dataset.get_mask(i), 128, 1, cv2.THRESH_BINARY)[1] for i, item in enumerate(mask_dataset)]
gt_masks = MaskDataset("datasets/qsd2_w1")

k = 10

sims = calc_similarities(dist.canberra, DB, QS2, True)
tops = get_tops(sims, k)
mapAtK = metrics.mapk(groundTruth, tops, k)

print(str(tops[0]))
print(str(tops[1]))
print(str(tops[2]))

print("Map@k is " + str(mapAtK))

mask_metrics = get_mask_metrics(predicted_masks, gt_masks, True)

print("Precision: " + str(mask_metrics["precision"]))
print("Recall: " + str(mask_metrics["recall"]))
print("F1-score: " + str(mask_metrics["f1_score"]))

#If you want to see the computed masks set this if to True
if False:
    predicted_masks = [mask_dataset.get_mask(i) for i, item in enumerate(mask_dataset)]
    for i, mask in enumerate(predicted_masks):
        cv2.imshow('ImageWindow', mask)
        cv2.waitKey()
