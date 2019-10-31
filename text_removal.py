import dataset
import glob
import numpy as np
import cv2
import utils
import math

import scipy.signal as sci
import distance as dist
import pytesseract
from tqdm.auto import tqdm


class text_remover(dataset.Dataset):
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))
        self.cache = [cv2.imread(path) for path in tqdm(self.paths)]

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def getpoints2(im, mask):
    mask_idxs = np.where(mask == 1)
    xs = mask_idxs[0]
    ys = mask_idxs[1]
    im_masked = im[min(xs):max(xs), min(ys):max(ys)]
    return getpoints2(im_masked)


def getpoints2(im, file_name):
    # ___GET SMALL BOUNDING BOX WITH NO FALSE POSITIVES___

    grad_neighb_divider = 2
    sobel_x_thresh = 25

    blur = cv2.GaussianBlur(im, (15, 15), 0)

    sobel_x_op = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x_op = np.abs(sobel_x_op)
    sobel_x_op = (sobel_x_op / np.amax(sobel_x_op) * 255).astype("uint8")

    h, l, s = cv2.split(cv2.cvtColor(sobel_x_op, cv2.COLOR_BGR2HLS))
    s_f = s.astype("float16")

    sobel_x_mod = cv2.cvtColor(sobel_x_op, cv2.COLOR_BGR2GRAY).astype("float16")
    sobel_x_mod -= s_f
    sobel_x_mod[sobel_x_mod < 0] = 0

    sobel_x_mod = sobel_x_mod / np.amax(sobel_x_mod) * 255
    sobel_x_mod = sobel_x_mod.astype("uint8")
    sobel_x_mod[sobel_x_mod < sobel_x_thresh] = 0

    start_x = round(13 * im.shape[1] / 32)
    end_x = round(19 * im.shape[1] / 32)

    min_y = 0
    min_value = 99999999999999

    for y in range(round(im.shape[0] / 4 - 4)):
        j_p = round(3 * im.shape[0] / 4) + y
        grad_devia = 0
        grad_x_neighb = 0

        mean_b = np.mean(blur[j_p][start_x:end_x])
        for x in range(start_x, end_x):
            grad_devia += abs(mean_b - blur[j_p, x])

        for x in range(start_x, end_x):
            grad_x_neighb += sobel_x_mod[j_p + 1, x]
            grad_x_neighb += sobel_x_mod[j_p + 2, x]
            grad_x_neighb += sobel_x_mod[j_p + 3, x]

        grad_x_neighb = 0.001 if grad_x_neighb == 0 else grad_x_neighb / grad_neighb_divider
        new_value = sum(grad_devia) / grad_x_neighb

        if new_value < min_value:
            min_value = new_value
            min_y = j_p

        ##########

        j_m = round(im.shape[0] / 4) - y
        grad_devia = 0
        grad_x_neighb = 0

        mean_b = np.mean(blur[j_m][start_x:end_x])
        for x in range(start_x, end_x):
            grad_devia += abs(mean_b - blur[j_m, x])

        for x in range(start_x, end_x):
            grad_x_neighb += sobel_x_mod[j_m + 1, x]
            grad_x_neighb += sobel_x_mod[j_m + 2, x]
            grad_x_neighb += sobel_x_mod[j_m + 3, x]
            grad_x_neighb += sobel_x_mod[j_m + 4, x]
            grad_x_neighb += sobel_x_mod[j_m + 5, x]
            grad_x_neighb += sobel_x_mod[j_m + 6, x]
            grad_x_neighb += sobel_x_mod[j_m + 7, x]

        grad_x_neighb = 0.001 if grad_x_neighb == 0 else grad_x_neighb / grad_neighb_divider
        new_value = sum(grad_devia) - grad_x_neighb

        if new_value < min_value:
            min_value = new_value
            min_y = j_m

    boundingxy_initial = [start_x, min_y, end_x, min_y + 10]
    boundingxy = np.copy(boundingxy_initial)

    # ___EXPAND BOUNDING BOX TO GET LESS FALSE NEGATIVES___

    # Otsu's thresholding.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    gray_b = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    imx = np.copy(gray_b[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]])

    ima, th2 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Getting info about if the image needs to be opened / closed to remove text from textbox.
    minv = np.min(th2)
    maxv = np.max(th2)
    meanv = np.mean(th2)

    diffmin = abs(meanv - minv)
    diffmax = abs(meanv - maxv)

    if diffmin < diffmax:
        imb, th2 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Getting good kernel size for opening / closing to remove text from textbox.
    maxkernel = 3

    for i in range(0, imx.shape[0]):
        for j in range(0, imx.shape[1]):
            if th2[i, j] == 0:
                for v in range(5, 13, 2):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (v, v))
                    w = math.floor(v / 2)
                    roi = th2[i - w : i + w + 1, j - w : j + w + 1]

                    if roi.shape == kernel.shape:
                        k = (roi * kernel).sum()
                        if k == 0:
                            if v > maxkernel:
                                maxkernel = v

    ksize = maxkernel + 12
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    if diffmin < diffmax:
        gray_b = cv2.morphologyEx(gray_b, cv2.MORPH_OPEN, kernel)
    else:
        gray_b = cv2.morphologyEx(gray_b, cv2.MORPH_CLOSE, kernel)

    # Getting gradients from opening / closing.
    sobel_x_op = cv2.Sobel(gray_b, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x_op = np.abs(sobel_x_op)
    sobel_x_op = (sobel_x_op / np.amax(sobel_x_op) * 255).astype("uint8")

    sobel_y_op = cv2.Sobel(gray_b, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y_op = np.abs(sobel_y_op)
    sobel_y_op = (sobel_y_op / np.amax(sobel_y_op) * 255).astype("uint8")

    sobel_y_op[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]] = 0
    sobel_x_op[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]] = 0

    # Expanding bbox until high gradients are found (border of text box).
    thresh = 20

    def moveboundy(range_m, b1, b2, b3, action, diff=-1):
        for i in range(b1, range_m, action):
            broken = False
            for j in range(b2, b3):
                if sobel_y_op[i, j] > thresh:
                    broken = True
                    break
            if broken:
                break
            else:
                b1 += action
                diff = 1
        if diff == 1:
            # Reduce temporaly bounding box in y direction to allow growth in x direction.
            b1 -= 5 * action
        return b1, diff

    def moveboundx(b1, b2, b3, range_m, action):
        for j in range(b1, range_m, action):
            broken = False
            for i in range(b2, b3):
                if sobel_x_op[i, j] > thresh:
                    broken = True
                    break
            if broken:
                break
            else:
                b1 += action
        return b1

    boundingxy[1], diff1 = moveboundy(0, boundingxy[1], boundingxy[0], boundingxy[2], -1, 0)
    boundingxy[3], diff2 = moveboundy(gray_b.shape[0], boundingxy[3], boundingxy[0], boundingxy[2], 1, 0)
    boundingxy[0] = moveboundx(boundingxy[0], boundingxy[1], boundingxy[3], 0, -1)
    boundingxy[2] = moveboundx(boundingxy[2], boundingxy[1], boundingxy[3], im.shape[1], 1)

    if diff1 == 1:
        boundingxy[1] -= 5

    if diff2 == 1:
        boundingxy[3] += 5

    # Drawing initial and expanded bbox.

    drawing = np.copy(im)
    cv2.rectangle(
        drawing,
        (boundingxy[0], boundingxy[1]),
        (boundingxy[2], boundingxy[3]),
        (255, 0, 0),
        2,
    )

    cv2.rectangle(
        drawing,
        (boundingxy_initial[0], boundingxy_initial[1]),
        (boundingxy_initial[2], boundingxy_initial[3]),
        (0, 255, 0),
        2,
    )

    imx = np.copy(gray[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]])
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(imx)
    with open("outputs/text/" + f"{file_name:05d}" + ".txt", "w") as f:
        f.write(text)


    mask = np.ones((im.shape[0], im.shape[1]))
    cv2.rectangle(mask, (boundingxy[0], boundingxy[1]), (boundingxy[2], boundingxy[3]), 0, -1)

    if False:
        cv2.imshow("Drawing", utils.resize(drawing, 50))
        cv2.imshow("Sobel x", utils.resize(sobel_x_mod, 50))
        cv2.waitKey(0)

    class Result:
        def __init__(self, boundingxy, drawing, text, mask):
            self.boundingxy = boundingxy
            self.drawing = drawing
            self.text = text
            self.mask = mask

    return Result(boundingxy, drawing, text, mask)