import cv2
from skimage import feature
import numpy as np

def get_lbp(img, blocks_h=0, blocks_v=0, mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        new_mask = mask[:, :] / 255
    else:
        new_mask = np.ones((img.shape[0], img.shape[1]))
    if blocks_h == 0 and blocks_v == 0:
        lbp = feature.local_binary_pattern(gray, 8, 1, method="default")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, 256),
                                 range=(0, 256), weights=new_mask.ravel())
        return hist

    hists = []
    if mask is not None:
        mask = mask[:, :] / 255
    else:
        mask = None

    height = gray.shape[0]
    width = gray.shape[1]

    block_v = height // blocks_v
    block_h = width // blocks_h

    for iter_i, i in enumerate(range(0, height, block_v)):
        if iter_i >= blocks_v:
            break
        for iter_j, j in enumerate(range(0, width, block_h)):
            if iter_j >= blocks_h:
                break
            block = gray[i:i + block_v, j:j + block_h]

            submask = mask[i:i + block_v, j:j + block_h] if mask is not None else None

            lbp = np.float32(feature.local_binary_pattern(block, 8, 1, method='default'))

            hist = cv2.calcHist([lbp], [0], submask, [32], [0, 256])
            cv2.normalize(hist, hist)
            hists.append(hist)
    return hists