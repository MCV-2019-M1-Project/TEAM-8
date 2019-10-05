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
    """
        Calculates the histogram of the images
        and applies a mask on from RGB to HLS on the Histogram calculation
    """


    def __init__(self, *args, caching=True, **kwargs):
        self.caching = caching
        if caching:
            self.cache = dict()
        super().__init__(*args, **kwargs)

    @staticmethod
    def calc_hist(img):
        def calc_mask():
            result = np.zeros(img.shape[:2], dtype="uint8")

            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            hls_split = cv2.split(hls)

            lu = np.asarray(hls_split[1])
            sa = np.asarray(hls_split[2])

            sat_thresh = 100
            lum_tresh = 75

            result[(sa > sat_thresh) & (lu < lum_tresh)] = 1
            return result

        """
        if background should be removed or not
        """
        if True:
            mask = calc_mask()
        else:
            mask = None

        return np.array(
            [
                cv2.calcHist([img], [0], mask, [256], [0, 256]),
                cv2.calcHist([img], [1], mask, [256], [0, 256]),
                cv2.calcHist([img], [2], mask, [256], [0, 256]),
            ]
        )

    def _calculate(self, idx):
        self.cache[idx] = self.calc_hist(super().__getitem__(idx))
        return self.cache[idx]

    def __getitem__(self, idx):
        if self.caching:
            return self.cache[idx] if idx in self.cache else self._calculate(idx)
        return self.calc_hist(super().__getitem__(idx))
