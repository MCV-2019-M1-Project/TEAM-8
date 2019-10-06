import glob

import numpy as np
import cv2


class Dataset:
    def __init__(self, path, mask):
        self.paths = glob.glob(f"{path}/*.jpg")

    def __getitem__(self, idx):
        # return cv2.imread(self.paths[idx],cv2.IMREAD_GRAYSCALE)
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
        print(args)
        self.masking = args[1]

    def calc_hist(self, img):
        def calc_mask():

            im = cv2.GaussianBlur(img, (5, 5), 0)

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            i, j = np.where(thresh == 255)
            y1 = i[0]  # fist point
            x1 = j[0]
            y4 = i[-1]  # last point
            x4 = j[-1]

            k, d = np.where(thresh[:, 0:100] == 255)
            y2 = k[0]  # second point
            x2 = d[0]
            y3 = k[-1]  # third point
            x3 = d[-1]

            points = np.array([[11, 13],
                               [14, 16],
                               [17, 11],
                               [12, 15]]).astype('int32')

            points[1] = (j[0], i[0])
            points[2] = (d[0], k[0])
            points[3] = (d[-1], k[-1])
            points[0] = (j[-1], i[-1])

            image_countours = cv2.fillPoly(thresh, np.int32([points]), (255, 255, 255), 8, 0, None)

            return image_countours

        if self.masking:
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


class MaskDataset:
    def __init__(self, path):
        self.paths = glob.glob(f"{path}/*.png")

    def __getitem__(self, idx):
        mask = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[1]

    def __len__(self):
        return len(self.paths)


