from os.path import Path

import numpy as np
import cv2
from methodtools import lru_cache

from utils import binsearch, normalize_hist
import text_removal


class Dataset:
    """
    TODO:
        - backup:
            - load idx - [stage_name, {**new_items}]
            - update idx
            - load all

        pyTables - complicated
        pickle - to many files, or can't load by index
        numpy.savez - everything has to be an array

        - compute stage by stage
    """

    pipeline = tuple()

    def __init__(self, path, extension="jpg", backup="", start_from=None):
        # TODO: allow many extensions
        self.paths = sorted(Path(path).glob("*.{extension}"))
        self.start_from = start_from
        if backup:
            self._init_backup(backup)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self._getitem_cached(idx)

    def _init_backup(self, backup_path):
        # TODO actual backup
        pass

    @lru_cache()
    def _getitem_cached(self, idx):
        data = self._load(idx)
        return self._finish_processing(data)

    def _load(self, idx):
        # TODO try to load from backup
        data = {"_dataset": self, "idx": idx, "path": self.paths[idx], "_completed": []}
        return data

    def _finish_processing(self, data, end_when_data_has=None):
        to_run = self._get_remaining_stages(data["_completed"])
        for stage_cls in to_run:
            stage_instance = stage_cls(data)
            try:
                new_items = self._apply_stage(stage_instance, data)
                self._save(data["idx"], new_items, stage_cls.__name__)
            except Exception:
                print(f"Failed in {stage_cls.__name__}")
                raise

    def save(self, idx):
        # TODO:
        pass

    def _get_remaining_stages(self, completed):
        try:
            first_difference = list(
                map(lambda c, s: c == s.__name__, zip(completed, self.pipeline))
            ).index(True)
        except ValueError:
            return ()
        return self.pipeline[first_difference:]

    def _apply_stage(self, stage, data):
        if getattr(stage, "validate", None) is not None:
            assert stage.validate(data), "Validation failed"
        if getattr(stage, "pre_apply", None) is not None:
            stage.pre_apply(data)
        new_items = stage.apply(data)
        if new_items:
            data.update(new_items)
        if getattr(stage, "post_apply", None) is not None:
            stage.post_apply(data)

    def assequence(self, of=None):
        if of is not None:
            if isinstance(of, str):
                return ({of: self.__getitem__(idx)[of]} for idx in range(len(self)))
            of = set(of)
            return (
                {
                    (key, value)
                    for key, value in self.__getitem__(idx).items()
                    if key in of
                }
                for idx in range(len(self))
            )
        return (self.__getitem__(idx) for idx in range(len(self)))


class BaseTransformMixin:
    def validate(self, data):
        return all(
            keyword in data for keyword in getattr(self, "required_keywords", [])
        )


class LoadImg(BaseTransformMixin):
    required_keywords = ("path",)

    def apply(self, data):
        img = cv2.imread(str(data["path"]))
        return {"img": img}


class LoadMask(BaseTransformMixin):
    required_keywords = ("path",)

    @classmethod
    def with_mask_ext(cls, ext):
        cls.ext = ext

    def apply(self, data):
        mask_path = data["path"].with_suffix(".png")
        mask = cv2.imread(str(mask_path))
        return {"mask": mask}


class MakeBBox(BaseTransformMixin):
    required_keywords = ("img",)

    def apply(self, data):
        img = data["img"]
        result = text_removal.getpoints2(img)
        bbox_coords = result.boundingxy
        base_mask = np.ones_like(img[:, :, 0])
        base_mask[bbox_coords[1] : bbox_coords[3], bbox_coords[0] : bbox_coords[2]] = 0
        return {"bbox_mask": np.uint8(base_mask) * 255, "bbox_coords": bbox_coords}


class MakeHist(BaseTransformMixin):
    required_keywords = ("img", "mask")

    def apply(self, data):
        img, mask = data["img"], data["mask"]
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
