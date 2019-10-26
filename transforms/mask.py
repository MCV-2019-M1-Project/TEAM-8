# This whole module is a one big TODO
import cv2
import numpy as np

from utils import binsearch


class SplitImages:
    required_keywords = ("img")

    def apply(self, data):
        self.make_base_mask()
        return {"split_masks": self.split_masks()}


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


# class Whatever:
#     def get_mask(self, idx):
#         if self.masking or self.bbox:
#             img = Dataset.__getitem__(self, idx)
#             mask = np.ones_like(img[:, :, 0]).astype(bool)
#             if self.masking:
#                 mask = mask & Mask(img).get_mask().astype(bool)
#             if self.bbox:
#                 mask = mask & BBox().get_bbox(img).astype(bool)
#             mask = mask.astype(int)
#             mask[mask != 0] = 255
#             return mask
#         return None

#     def get_masks(self, idx):
#         img = Dataset.__getitem__(self, idx)
#         bbox = BBox().get_bbox(img)
#         for mask in Splitter(img):
#             res = mask + bbox
#             res[res != 0] = 255
#             yield res


# class MultiHistDataset(HistDataset):
#     def __init__(self, *args, **kwargs):a
#         super().__init__(*args, **kwargs)

#     def normalize(self, hists):
#         return [normalize_hist(h) for h in hists]

#     def calc_hist(self, idx):
#         img = Dataset.__getitem__(self, idx)
#         return [self._calc_hist(img, mask) for mask in self.get_masks(idx)]


# class MaskDataset:
#     def __init__(self, path):
#         self.paths = sorted(glob.glob(f"{path}/*.png"))

#     def __getitem__(self, idx):
#         mask = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
#         return cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)[1]

#     def __len__(self):
#         return len(self.paths)
