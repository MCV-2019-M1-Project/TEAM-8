"""
Example usage: python run.py task2
"""

import numpy as np
import cv2
import fire
import ml_metrics as metrics
from tqdm.auto import tqdm


import text_removal
from utils import get_groundtruth, get_mean_IoU, normalize_hist
from dataset import Dataset, MaskDataset, HistDataset
import distance as dist

from utils import calc_similarities, get_tops, get_mask_metrics


def find_img_corresp(QS, GT, DB, k):
    similarieties = calc_similarities(dist.canberra, DB, QS, True)
    tops = get_tops(similarieties, k)
    mapAtK = metrics.mapk(GT, tops, k)
    print("Map@k is " + str(mapAtK))


def eval_masks(QS, MS_GT):
    if len(QS) != len(MS_GT):
        raise ValueError("Query set size doesn't match ground truth size")
    mask_dataset = Dataset("datasets/qsd2_w1", masking=True)
    print("Predicting masks")
    predicted_masks = [
        mask_dataset.get_mask(i) // 255
        for i in tqdm(range(len(QS)))
    ]
    print("Calculating mask metrics")
    mask_metrics = get_mask_metrics(predicted_masks, MS_GT, True)

    print("Precision: " + str(mask_metrics["precision"]))
    print("Recall: " + str(mask_metrics["recall"]))
    print("F1-score: " + str(mask_metrics["f1_score"]))


class Solution:
    def __init__(
        self,
        DDBB="datasets/DDBB",
        QSD1_W1="datasets/qsd1_w1",
        QSD2_W1="datasets/qsd2_w1",
        QSD1_W2="datasets/qsd1_w2",
        QSD2_W2="datasets/qsd2_w2",
    ):
        self.QSD1_W1 = QSD1_W1
        self.QSD2_W1 = QSD2_W1
        self.QSD1_W2 = QSD1_W2
        self.QSD2_W2 = QSD2_W2
        self.DDBB = DDBB

    def task2(self, k=10):
        QS2 = HistDataset(self.QSD2_W1, masking=True, multires=4)
        GT = get_groundtruth("datasets/qsd2_w1/gt_corresps.pkl")
        print(f"Computing normalized histograms for {self.DDBB}")
        DB = [
            normalize_hist(db_hist)
            for db_hist in tqdm(HistDataset(self.DDBB, masking=False, multires=4))
        ]
        print("Analyzing QS2")
        find_img_corresp(QS2, GT, DB, k)

    def eval_masks(self):
        QS = Dataset(self.QSD2_W1, masking=True)
        GT = MaskDataset(self.QSD2_W1)
        eval_masks(QS, GT)

    def eddie(self):
        QS1 = [
            text_removal.getpoints2(im)
            for im in text_removal.text_remover(self.QSD1_W2)
        ]
        boundingxys = [element.boundingxy for element in QS1]
        drawings = [element.drawing for element in QS1]

        gt = np.asarray(get_groundtruth(f"{self.QSD1_W2}/text_boxes.pkl")).squeeze()
        mean_IoU = get_mean_IoU(gt, boundingxys)

        print("Mean Intersection over Union: ", mean_IoU)

        for im in range(len(drawings)):
            cv2.imwrite("outputs/" + str(im) + ".png", drawings[im])


if __name__ == "__main__":
    fire.Fire(Solution)
