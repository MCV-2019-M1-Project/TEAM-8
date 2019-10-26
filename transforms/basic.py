import cv2
import numpy as np

from mixins import BaseTransformMixin, MultiTransformMixin
from utils import show_img
from text_removal import getpoints2


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


class ShowImg(BaseTransformMixin):
    @classmethod
    def img_keyword(cls, keyword):
        cls.keyword = keyword

    def apply(self, data):
        show_img(data[self.keyword])


class MakeBBox(BaseTransformMixin, MultiTransformMixin):
    # TODO multi mixin
    required_keywords = ("img",)
    multi_key = "split_mask"

    def apply_single(self, data, split_mask):
        img = data["img"]

        result = getpoints2(img)
        bbox_coords = result.boundingxy
        base_mask = np.ones_like(img[:, :, 0])
        base_mask[bbox_coords[1] : bbox_coords[3], bbox_coords[0] : bbox_coords[2]] = 0
        return {"bbox_mask": np.uint8(base_mask) * 255, "bbox_coords": bbox_coords}
