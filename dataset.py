import glob

import numpy as np
import cv2


class Dataset:
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))
        print(len(self.paths))

    def __getitem__(self, idx):
        return cv2.imread(self.paths[idx])

    def __len__(self):
        return len(self.paths)


class HistDataset(Dataset):
    """
        Calculates the histogram of the images
        and applies a mask on from RGB to HLS on the Histogram calculation
    """

    def __init__(self, *args, caching=True, masking=False, **kwargs):
        self.caching = caching
        self.masking = masking
        if caching:
            self.cache = dict()
        super().__init__(*args, **kwargs)

    def calc_mask(self, img):
        """
        This is to remove the background it basically searches for the key points of the rectangle and removes the background.

        It just works fine with this particular scenario.

        P2  P1

        P3  P0

        """
        im = cv2.GaussianBlur(img, (5, 5), 0)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        i, j = np.where(unknown == 255)
        k, d = np.where(thresh[:, 0:100] == 255)

        points = np.array(
            [(j[-1], i[-1]), (j[0], i[0]), (d[0], k[0]), (d[-1], k[-1])]
        )

        image_countours = cv2.fillPoly(
            unknown, np.int32([points]), (255, 255, 255), 8, 0, None
        )

        return image_countours

    def calc_hist(self, img):

        mask = None if not self.masking else self.calc_mask(img)

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

    def get_mask(self, idx):
        return self.calc_mask(super().__getitem__(idx))


class MaskDataset:
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.png"))

    def __getitem__(self, idx):
        mask = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[1]

    def __len__(self):
        return len(self.paths)
