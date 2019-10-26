import dataset
import glob
import numpy as np
import cv2
import utils
import math

import scipy.signal as sci
import distance as dist
import pytesseract


class text_remover(dataset.Dataset):
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))

    def __getitem__(self, idx):
        return super().__getitem__(idx)


def getpoints(im):

    ret, thresh1 = cv2.threshold(im[:, :, 0], 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    sure_bg1 = cv2.erode(thresh1, kernel, iterations=3)

    ret, thresh2 = cv2.threshold(im[:, :, 1], 200, 255, cv2.THRESH_BINARY)

    sure_bg2 = cv2.erode(thresh2, kernel, iterations=3)

    ret, thresh3 = cv2.threshold(im[:, :, 2], 200, 255, cv2.THRESH_BINARY)
    sure_bg3 = cv2.erode(thresh3, kernel, iterations=3)

    final = sure_bg1 - (sure_bg3 - sure_bg1 - sure_bg2)
    return final


def getpoints2(im):

    # ___GET Y INSIDE TEXT___

    grad_neighb_divider = 1000
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

    start_x = round(3 * im.shape[1] / 8)
    end_x = round(5 * im.shape[1] / 8)

    min_y = 0
    min_value = 99999999999999

    im_m = np.copy(im)

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
            grad_x_neighb += sobel_x_mod[j_m - 1, x]
            grad_x_neighb += sobel_x_mod[j_m - 2, x]
            grad_x_neighb += sobel_x_mod[j_m - 3, x]

        grad_x_neighb = 0.001 if grad_x_neighb == 0 else grad_x_neighb / grad_neighb_divider
        new_value = sum(grad_devia) / grad_x_neighb

        if new_value < min_value:
            min_value = new_value
            min_y = j_m

    pad = 3
    im_m[min_y - pad:min_y + pad, start_x:end_x] = [0, 0, 255]
    im_m[min_y, start_x:end_x] = [255, 0, 0]

    im_m[min_y - 10 - pad:min_y - 10 + pad, start_x:end_x] = [0, 255, 0]

    boundingxy = [start_x, min_y - 10, end_x, min_y]
    boundingxy2 = [start_x, min_y - 10, end_x, min_y]

    im_m = cv2.rectangle(
        im_m,
        (boundingxy[0], boundingxy[1]),
        (boundingxy[2], boundingxy[3]),
        (0, 0, 255),
        2,
    )

    # ___GET BOUNDARIES___

    # Otsu's thresholding
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    gray_b = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    imx = np.copy(gray_b[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]])

    ima, th2 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Getting info about if the image needs to be opened / closed to remove text from textbox
    minv = np.min(th2)
    maxv = np.max(th2)
    meanv = np.mean(th2)

    diffmin = abs(meanv - minv)
    diffmax = abs(meanv - maxv)

    if diffmin < diffmax:
        imb, th2 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Getting good kernel size for opening / closing to remove text from textbox
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

    ksize = maxkernel + 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    if diffmin < diffmax:
        gray_b = cv2.morphologyEx(gray_b, cv2.MORPH_OPEN, kernel)
    else:
        gray_b = cv2.morphologyEx(gray_b, cv2.MORPH_CLOSE, kernel)

    # Getting gradients from opening / closing
    sobel_x_op = cv2.Sobel(gray_b, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x_op = np.abs(sobel_x_op)
    sobel_x_op = (sobel_x_op / np.amax(sobel_x_op) * 255).astype("uint8")

    sobel_y_op = cv2.Sobel(gray_b, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y_op = np.abs(sobel_y_op)
    sobel_y_op = (sobel_y_op / np.amax(sobel_y_op) * 255).astype("uint8")

    sobel_y_op[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]] = 0
    sobel_x_op[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]] = 0

    # Expanding bbox until high gradients are found (border of text box)
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

    boundingxy[1], diff1 = moveboundy(
        0, boundingxy[1], boundingxy[0], boundingxy[2], -1, 0
    )
    boundingxy[3], diff2 = moveboundy(
        gray_b.shape[0], boundingxy[3], boundingxy[0], boundingxy[2], 1, 0
    )
    boundingxy[0] = moveboundx(boundingxy[0], boundingxy[1], boundingxy[3], 0, -1)
    boundingxy[2] = moveboundx(boundingxy[2], boundingxy[1], boundingxy[3], im.shape[1], 1)

    if diff1 == 1:
        boundingxy[1] -= 5

    if diff2 == 1:
        boundingxy[3] += 5

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
        (boundingxy2[0], boundingxy2[1]),
        (boundingxy2[2], boundingxy2[3]),
        (0, 255, 0),
        2,
    )

    imx = np.copy(gray[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]])
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(imx)

    if False:
        cv2.imshow("Drawing", utils.resize(drawing, 50))
        #cv2.imshow("Sobel x", utils.resize(sobel_x_op, 50))
        #cv2.imshow("Sobel y", utils.resize(sobel_y_op, 50))
        cv2.waitKey(0)

    class Result:
        def __init__(self, boundingxy, drawing, text):
            self.boundingxy = boundingxy
            self.drawing = drawing
            self.text = text

    return Result(boundingxy, drawing, text)


def getpoints3(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imagehigh = gray > 200
    imagelow = gray < 50
    imager = im[:, :, 0]
    imageg = im[:, :, 1]
    imageb = im[:, :, 2]
    output = np.equal(imager, imageg, dtype=int)
    output2 = np.equal(imager, imageb, dtype=int)
    output1 = np.equal(imageb, imageg, dtype=int)
    output3 = output & output2
    output4 = output3 & output1
    equalhigh = output4 & imagehigh
    equallow = output4 & imagelow

    kernel = np.ones((int((im.shape[0]) / 350), int((im.shape[1]) / 20)), np.uint8)
    equallow = 255 * equallow
    equallow = equallow.astype(np.uint8)
    equalhigh = 255 * equalhigh
    equalhigh = equalhigh.astype(np.uint8)

    equalhigh = cv2.erode(equalhigh, kernel, iterations=1)
    equalhigh = cv2.dilate(equalhigh, kernel, iterations=6)
    equalhigh = cv2.erode(equalhigh, kernel, iterations=7)
    if equalhigh.any() > 0:
        print("imatge blanca")
        # TODO: Probar amb el threshold aqui,
        # ja que saps si es blanca o negre amb un 1/30 de error.
        # TODO(Stas): Please comment in English
        final = 255 * imagehigh
        final = final.astype(np.uint8)
    else:
        print("imatge negre")
        final = None

    return final
