import glob

import numpy as np
import cv2
import text_removal


class Dataset:
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))
        self.data = [cv2.imread(path) for path in self.paths]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.paths)


class HistDataset(Dataset):
    """
        Calculates the histogram of the images
        and applies a mask on from RGB to HLS on the Histogram calculation
    """

    def __init__(self, *args, caching=True, masking=False, bbox=False, dimensions=1, block=0, multires=0, **kwargs):
        self.caching = caching
        self.masking = masking
        self.bbox = bbox
        self.dimensions = dimensions
        self.block = block
        self.multires = multires

        if caching:
            self.cache = dict()
        super().__init__(*args, **kwargs)

    def calc_mask(self, img):
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
        ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        combi = unknown + sure_fg
        combi[combi > 255] = 255

        i, j = np.where(combi == 255)
        k, d = np.where(combi[:, 0:150] == 255)

        points = np.array(
            [(j[-1], i[-1]), (j[0], i[0]), (d[0], k[0]), (d[-1], k[-1])]
        )

        image_countours = cv2.fillPoly(
            combi, np.int32([points]), (255, 255, 255), 8, 0, None
        )

        return image_countours

    def calc_bbox_as_mask(self, img):
        base_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        base_mask[:, :] = 1
        result = text_removal.getpoints2(img)
        bbox_coords = result.boundingxy
        base_mask[bbox_coords[1]:bbox_coords[3], bbox_coords[0]: bbox_coords[2]] = 0
        return base_mask

    def calc_hist(self, img):

        if self.masking:
            mask = self.calc_mask(img)
        elif self.bbox:
            mask = self.calc_bbox_as_mask(img)
        else:
            mask = None
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = hsv

        if self.dimensions == 2:
            hist = cv2.calcHist([hsv], [0, 1], mask, [180/8, 256/8], [0, 180, 0, 256])
            hist = hist / hist.sum(axis=-1, keepdims=True)
            hist[np.isnan(hist)] = 0
            onedhist = np.reshape(hist, [-1])
            return onedhist

        if self.dimensions == 3:
            hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist/hist.sum(axis=-1, keepdims=True)
            hist[np.isnan(hist)] = 0
            onedhist = np.reshape(hist, [-1])
            return onedhist

        if self.block >= 1:
            hists = []
            for i in range(3):
                hists.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
            for divisions in range(1, self.block+1):
                imgheight = img.shape[0]
                imgwidth = img.shape[1]

                y1 = 0
                M = imgheight // 2**divisions
                N = imgwidth // 2**divisions

                for y in range(0, imgheight, M):
                    for x in range(0, imgwidth, N):
                        y1 = y + M
                        x1 = x + N
                        for i in range(3):
                            submask = mask[y:y + M, x:x + N] if self.masking or self.bbox else None
                            hists.append(cv2.calcHist([img[y:y + M, x:x + N]], [i], submask, [256], [0, 256]))
            return hists

        if self.multires > 1:
            hists = []
            for i in range(3):
                hists.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
            for res in range(2, self.multires+1):
                imgheight = img.shape[0] // res
                imgwidth = img.shape[1] // res
                for i in range(3):
                    submask = cv2.resize(mask, (imgwidth, imgheight)) if self.masking or self.bbox else None
                    downscale = cv2.resize(img, (imgwidth, imgheight))
                    hists.append(cv2.calcHist([downscale], [i], submask, [256], [0, 256]))
            return hists

        return np.array(
            [
                cv2.calcHist([img], [0], mask, [256], [0, 256]),
                cv2.calcHist([img], [1], mask, [256], [0, 256]),
                cv2.calcHist([img], [2], mask, [256], [0, 256]),
                cv2.calcHist([hsv], [0], mask, [256], [0, 256]),
                cv2.calcHist([hsv], [1], mask, [256], [0, 256]),
                cv2.calcHist([hsv], [2], mask, [256], [0, 256]),
                cv2.calcHist([lab], [0], mask, [256], [0, 256]),
                cv2.calcHist([lab], [1], mask, [256], [0, 256]),
                cv2.calcHist([lab], [2], mask, [256], [0, 256]),
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

    def get_bbox(self, idx):
        return self.calc_bbox_as_mask(super().__getitem__(idx))


class MaskDataset:
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.png"))

    def __getitem__(self, idx):
        mask = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[1]

    def __len__(self):
        return len(self.paths)
