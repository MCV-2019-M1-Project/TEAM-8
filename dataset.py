import glob

import numpy as np
import cv2

from utils import binsearch, normalize_hist
import text_removal


class Mask:
    """
    Helper class to make a mask out of image
    Proposes various masks, run the tests and picks the best one.
    """

    def __init__(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.img = img
        self.std_thresh = 60 / 100
        self.overall_std = np.std(img)

        border_mask = np.zeros_like(img)
        border_mask[:2, :] = 1
        border_mask[-2:, :] = 1
        border_mask[:, -2:] = 1
        border_mask[:, :2] = 1
        border_mask = border_mask.astype(bool)
        self.borders_std = np.std(img[border_mask])
        self.borders_mean = np.mean(img[border_mask])

    def horizontal_mask(self, img, end, begin=0):
        if begin > img.shape[0]:
            raise IndexError
        res = np.zeros_like(img)
        res[begin:end, :] = 1
        return res.astype(bool)

    def check_std(self, img, mask):
        test_vals = img[mask]
        target_std = self.overall_std * self.std_thresh
        return np.std(test_vals) < target_std

    def check_mean(self, img, mask):
        test_vals = img[mask]
        return (
            True
            if test_vals.size == 0
            else abs(np.mean(test_vals) - self.borders_mean) < 2 * self.borders_std
        )

    def check_with_border(self, img, mask, thresh=1.8):
        test_vals = img[mask]
        target_std = self.borders_std * thresh
        return True if test_vals.size == 0 else np.std(test_vals) < target_std

    def check_tests(self, img, maybe_mask):
        return self.check_std(img, maybe_mask)  # & self.check_mean(img, maybe_mask)

    def get_one_edge(self, rotation):
        rimg = np.rot90(self.img, rotation)
        beg, end = 0, rimg.shape[0] // 2
        res = binsearch(
            beg,
            end,
            lambda mid: self.check_tests(rimg, self.horizontal_mask(rimg, mid)),
        )
        res = binsearch(
            beg,
            res,
            lambda mid: self.check_with_border(
                rimg, self.horizontal_mask(rimg, mid, mid - 5)
            ),
        )
        return np.rot90(self.horizontal_mask(rimg, res), 4 - rotation), res

    def get_middle(self, img, begin, end, step=None):
        rimg = np.rot90(img, 1)
        step = step or rimg.shape[0] // 20
        for beg in range(begin, end - step, step):
            tmp_mask = self.horizontal_mask(rimg, beg + step, beg)
            if self.check_with_border(rimg, tmp_mask, 1) and self.check_std(
                rimg, tmp_mask
            ):
                return beg + int(step / 2)
        return None

    def get_mask_single(self, numbers=False):
        masks = [self.get_one_edge(rot) for rot in range(4)]
        res = masks[0][0]
        cutpts = []
        for mask, cutpt in masks:
            res = res | mask
            cutpts.append(cutpt)
        res = ~res
        res = np.uint8(res) * 255
        return (res, cutpts) if numbers else res

    def get_mask(self):
        m, cut = self.get_mask_single(True)
        _, q, _, p = cut
        q = self.img.shape[1] - q
        x = self.get_middle(self.img, p, q)
        if x is not None:
            x = self.img.shape[1] - x
            m1 = Mask(self.img[:, :x]).get_mask_single()
            m2 = Mask(self.img[:, x:]).get_mask_single()
            res = np.concatenate((m1, m2), axis=1)
            return res
        return m


class Splitter:
    def __init__(self, img):
        self.img = img
        mask = Mask(img)
        self.single_mask, cut = mask.get_mask_single(True)
        _, q, _, p = cut
        q = self.img.shape[1] - q
        self.x = mask.get_middle(self.img, p, q)

    def __iter__(self):
        x = self.x
        if x is None:
            yield self.single_mask
        else:
            left = np.zeros_like(self.img[:, :, 0])
            left[:, :x] += Mask(self.img[:, :x]).get_mask_single()
            yield left
            right = np.zeros_like(self.img[:, :, 0])
            right[:, x:] += Mask(self.img[:, x:]).get_mask_single()
            yield right


class BBox:
    def get_bbox(self, img):
        base_mask = np.ones_like(img[:, :, 0])
        result = text_removal.getpoints2(img)
        bbox_coords = result.boundingxy
        base_mask[bbox_coords[1] : bbox_coords[3], bbox_coords[0] : bbox_coords[2]] = 0
        return np.uint8(base_mask) * 255

    def get_bbox_cords(self, img):
        result = text_removal.getpoints2(img)
        bbox_coords = result.boundingxy
        return bbox_coords


class Dataset:
    def __init__(self, path, masking=False, bbox=False):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))
        self.masking = masking
        self.bbox = bbox
        self.cache = [cv2.imread(path) for path in self.paths]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.paths)

    def get_mask(self, idx):
        if self.masking or self.bbox:
            img = Dataset.__getitem__(self, idx)
            mask = np.ones_like(img[:, :, 0]).astype(bool)
            if self.masking:
                mask = mask & Mask(img).get_mask().astype(bool)
            if self.bbox:
                mask = mask & BBox().get_bbox(img).astype(bool)
            mask = mask.astype(int)
            mask[mask != 0] = 255
            return mask
        return None

    def get_masks(self, idx):
        img = Dataset.__getitem__(self, idx)
        bbox = BBox().get_bbox(img)
        for mask in Splitter(img):
            res = mask + bbox
            res[res != 0] = 255
            yield res


class HistDataset(Dataset):
    """
        Calculates the histogram of the images
        and applies a mask on from RGB to HLS on the Histogram calculation
    """

    def __init__(
        self, *args, caching=True, dimensions=1, block=0, multires=0, **kwargs
    ):
        self.caching = caching
        self.dimensions = dimensions
        self.block = block
        self.multires = multires

        if caching:
            self.cache = dict()
        super().__init__(*args, **kwargs)

    def calc_hist(self, idx):
        img = super().__getitem__(idx)
        self.mask = self.get_mask(idx)
        return self._calc_hist(img, self.mask)

    def _calc_hist(self, img, mask):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = hsv

        if self.dimensions == 2:
            hist = cv2.calcHist(
                [hsv], [0, 1], mask, [180 / 8, 256 / 8], [0, 180, 0, 256]
            )
            hist = hist / hist.sum(axis=-1, keepdims=True)
            hist[np.isnan(hist)] = 0
            onedhist = np.reshape(hist, [-1])
            return onedhist

        if self.dimensions == 3:
            hist = cv2.calcHist(
                [img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256]
            )
            hist = hist / hist.sum(axis=-1, keepdims=True)
            hist[np.isnan(hist)] = 0
            onedhist = np.reshape(hist, [-1])
            return onedhist

        if self.block >= 1:
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
                            submask = mask[y : y + M, x : x + N] if self.masking else None
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

        if self.multires > 1:
            hists = []
            for i in range(3):
                hists.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
            for res in range(2, self.multires + 1):
                imgheight = img.shape[0] // res
                imgwidth = img.shape[1] // res
                for i in range(3):
                    submask = (
                        cv2.resize(mask, (imgwidth, imgheight)) if self.masking else None
                    )
                    downscale = cv2.resize(img, (imgwidth, imgheight))
                    hists.append(
                        cv2.calcHist([downscale], [i], submask, [256], [0, 256])
                    )
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
        self.cache[idx] = self.normalize(self.calc_hist(idx))
        return self.cache[idx]

    def normalize(self, hist):
        return normalize_hist(hist)

    def __getitem__(self, idx):
        if self.caching:
            return self.cache[idx] if idx in self.cache else self._calculate(idx)
        return self.normalize(self.calc_hist(super().__getitem__(idx)))


class MultiHistDataset(HistDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def normalize(self, hists):
        return [normalize_hist(h) for h in hists]

    def calc_hist(self, idx):
        img = Dataset.__getitem__(self, idx)
        return [self._calc_hist(img, mask) for mask in self.get_masks(idx)]


class MaskDataset:
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.png"))

    def __getitem__(self, idx):
        mask = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[1]

    def __len__(self):
        return len(self.paths)
