from itertools import chain

import numpy as np
from scipy.fftpack import dct

# import cv2


def compute_vector(img, slices=(5, 5), keep_size=10):
    def zig_zag_filter(patch, keep_size):
        # Array with distances from (0,0)
        way = np.ones_like(patch).cumsum()
        way = way + way.T
        # Get the keep_size first indices of patch
        way = np.argsort(way, axis=None)[:keep_size]
        return np.take_along_axis(patch, way, axis=None)

    if type(slices) == int:
        slices = (slices, slices)
    # Split images to slices parts, vertically and horizontally
    patches = np.array_split(img, slices[0])
    patches = [np.array_split(half_split, slices[1], axis=1) for half_split in patches]
    patches = chain.from_iterable(patches)
    # Compute the dct and take the first keep_size elements in zig-zag order
    patches = (dct(patch / 255, norm="ortho") for patch in patches)
    patches = (zig_zag_filter(patch, keep_size) for patch in patches)
    # Concatenate the results
    return np.array(list(patches)).reshape(-1)


# Example usage
# img = cv2.imread('datasets/qsd1_w1/00000.jpg')
# print(compute_vector(img, 10, 3))
