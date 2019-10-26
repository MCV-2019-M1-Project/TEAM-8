import cv2
import numpy as np

from mixins import BaseTransformMixin, MultiTransformMixin
from utils import normalize_hist


class MakeHist(BaseTransformMixin, MultiTransformMixin):
    required_keywords = ("img", "mask")
    multi_key = "mask"

    def apply_single(self, data, mask):
        img = data["img"]
        return normalize_hist(self.calculate_hist(img, mask))

    def caluculate_hist(self, img, mask):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        return np.array(
            [
                cv2.calcHist([img], [ch], mask, [256], [0, 256])
                for img in (hsv, lab)
                for ch in range(3)
            ]
        )


class MakeDimensionalHist(MakeHist):
    def calculate_hist(self, img, mask):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _2dparams = ([img], [0, 1], mask, [180 / 8, 256 / 8], [0, 180, 0, 256])
        _3dparams = ([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = (
            cv2.calcHist(*_2dparams)
            if self.dimension == 2
            else cv2.calcHist(*_3dparams)
        )
        hist = hist / hist.sum(axis=-1, keepdims=True)
        hist[np.isnan(hist)] = 0
        onedhist = np.reshape(hist, [-1])
        return onedhist


class MakeBlockHist(MakeHist):
    def calculate_hist(self, img, mask):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hists = []
        for i in range(3):
            hists.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
        for divisions in range(1, self.block + 1):
            imgheight = img.shape[0]
            imgwidth = img.shape[1]

            M = imgheight // 2 ** divisions
            N = imgwidth // 2 ** divisions

            for y in range(0, imgheight, M):
                for x in range(0, imgwidth, N):
                    for i in range(3):
                        submask = mask[y : y + M, x : x + N] if mask else None
                        hists.append(
                            cv2.calcHist(
                                [img[y : y + M, x : x + N]],
                                [i],
                                submask,
                                [256],
                                [0, 256],
                            )
                        )
        return hists


class MakeMultiResHist(MakeHist):
    def calculate_hist(self, img, mask):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hists = []
        for i in range(3):
            hists.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
        for res in range(2, self.multires + 1):
            imgheight = img.shape[0] // res
            imgwidth = img.shape[1] // res
            for i in range(3):
                submask = cv2.resize(mask, (imgwidth, imgheight)) if self.mask else None
                downscale = cv2.resize(img, (imgwidth, imgheight))
                hists.append(cv2.calcHist([downscale], [i], submask, [256], [0, 256]))
        return hists
