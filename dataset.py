import glob

import numpy as np
import cv2


class Dataset:
    def __init__(self, path):
        self.paths = glob.glob(f"{path}/*.jpg")

    def __getitem__(self, idx):
        return cv2.imread(self.paths[idx])

    def __len__(self):
        return len(self.paths)


class HistDataset(Dataset):
    def __init__(self, *args, caching=True, **kwargs):
        self.caching = caching
        if caching:
            self.cache = dict()
        super().__init__(*args, **kwargs)

    @staticmethod
    def calc_hist(img):
        return np.array(
            [
                cv2.calcHist([img], [0], None, [256], [0, 256]),
                cv2.calcHist([img], [1], None, [256], [0, 256]),
                cv2.calcHist([img], [2], None, [256], [0, 256]),
            ]
        )

    def _calculate(self, idx):
        self.cache[idx] = self.calc_hist(super().__getitem__(idx))
        return self.cache[idx]

    def __getitem__(self, idx):
        if self.caching:
            return self.cache[idx] if idx in self.cache else self._calculate(idx)
        return self.calc_hist(super().__getitem__(idx))
