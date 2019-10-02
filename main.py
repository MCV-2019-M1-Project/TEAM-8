import numpy as np
import cv2
import glob
from distance import euclidean


class Dataset:
    def __init__(self, path):
        self.paths = glob.glob("DDBB/Debug/*jpg")

    def __iter__(self):
        self

    def __next__(self):
        for i in len(self.paths):
            yield self[i]

    def __getitem__(self, idx):
        return cv2.imread(self.paths[idx])


def calc_hist(img):
    return np.array(
        [
            cv2.calcHist([img], [0], None, [256], [0, 256]),
            cv2.calcHist([img], [1], None, [256], [0, 256]),
            cv2.calcHist([img], [2], None, [256], [0, 256]),
        ]
    )


def compute_hists(dataset):
    return np.array([calc_hist(img) for img in dataset])


QS1 = Dataset("QS1")
DB = Dataset("DB")

db_hists = compute_hists(DB)


def calc_similarities(measure):
    def compute_one(img):
        hist = calc_hist(img)
        return [measure(hist, db_hist) for db_hist in db_hists]

    return np.array([compute_one(img) for img in QS1])


simils = calc_similarities(euclidean)


def get_tops(similarities):
    



