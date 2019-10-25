import dataset
import glob
import numpy as np
import cv2
import utils
import math

import scipy.signal as sci


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

    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x = np.abs(sobel_x)
    sobel_x = (sobel_x / np.amax(sobel_x) * 255).astype("uint8")

    h, l, s = cv2.split(cv2.cvtColor(sobel_x, cv2.COLOR_BGR2HLS))
    s_f = s.astype("float16")

    sobel_x_mod = cv2.cvtColor(sobel_x, cv2.COLOR_BGR2GRAY).astype("float16")

    sobel_x_mod -= s_f
    sobel_x_mod[sobel_x_mod < 0] = 0

    sobel_x_mod = sobel_x_mod / np.amax(sobel_x_mod) * 255
    sobel_x_mod = sobel_x_mod.astype("uint8")
    sobel_x_mod[sobel_x_mod < sobel_x_thresh] = 0

    start_x = round(im.shape[1] / 3)
    end_x = round(2 * im.shape[1] / 3)

    min_y = 0
    min_value = 99999999999999
    min_y2 = 0
    min_value2 = 99999999999999
    min_y3 = 0
    min_value3 = 99999999999999

    im_m = np.copy(im)

    for y in range(round(im.shape[0] / 4 - 4)):
        j_p = round(3 * im.shape[0] / 4) + y
        grad_devia = 0
        grad_x_neighb = 0

        mean_b = np.mean(blur[j_p][start_x:end_x])
        for x in range(start_x, end_x):
            grad_devia += abs(mean_b - blur[j_p, x])
            #im_m[j_p,x] = [255, 0, 0]

        for x in range(start_x, end_x):
            grad_x_neighb += sobel_x_mod[j_p + 1, x]
            grad_x_neighb += sobel_x_mod[j_p + 2, x]
            grad_x_neighb += sobel_x_mod[j_p + 3, x]

        grad_x_neighb = 0.001 if grad_x_neighb == 0 else grad_x_neighb / grad_neighb_divider
        new_value = sum(grad_devia) / grad_x_neighb

        if new_value < min_value:
            #print("NV: ", new_value, " Mean", sum(grad_devia), " X_Neighb", grad_x_neighb)
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

    im_m[min_y - 20 - pad:min_y - 20 + pad, start_x:end_x] = [0, 255, 0]

    boundingxy = [start_x, min_y - 20, end_x, min_y]
    print(boundingxy)

    im_m = cv2.rectangle(
        im_m,
        (boundingxy[0], boundingxy[1]),
        (boundingxy[2], boundingxy[3]),
        (0, 0, 255),
        2,
    )

    res = 50
    cv2.imshow("image", utils.resize(im_m, res))
    #cv2.imshow("sobel", utils.resize(sobel_x, res))
    #cv2.imshow("saturation", utils.resize(s, res))
    #cv2.imshow("joined", utils.resize(sobel_x_mod, res))

    # ___GET BOUNDARIES___

    # Betos Post-Processing from here on

    # Otsu's thresholding
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    imx = np.copy(gray[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]])

    cv2.imshow("a", imx)

    ima, th2 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print(th2)
    minv = np.min(th2)
    maxv = np.max(th2)
    meanv = np.mean(th2)

    diffmin = abs(meanv - minv)
    diffmax = abs(meanv - maxv)

    th3 = np.copy(th2)

    if diffmin < diffmax:
        imb, th3 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    maxkernel = 0

    for i in range(0, imx.shape[0]):
        for j in range(0, imx.shape[1]):
            if th3[i, j] == 0:
                for v in range(1, 15, 2):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (v, v))
                    w = math.floor(v / 2)
                    roi = th3[i - w : i + w + 1, j - w : j + w + 1]

                    if roi.shape == kernel.shape:
                        k = (roi * kernel).sum()
                        if k == 0:
                            if v > maxkernel:
                                maxkernel = v

    ksize = round(3 * maxkernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    if diffmin < diffmax:
        print("Background black")
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    else:
        print("Background white")
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    pad = 5000

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_x = np.abs(sobel_x)
    sobel_x = (sobel_x / np.amax(sobel_x) * 255).astype("uint8")

    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = np.abs(sobel_y)
    sobel_y = (sobel_y / np.amax(sobel_y) * 255).astype("uint8")

    sobel_y[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]] = 0
    sobel_x[boundingxy[1]:boundingxy[3], boundingxy[0]:boundingxy[2]] = 0

    thresh = 20

    def moveboundy(maxsize, b1, b2, b3, pad, action, diff=-1):
        for i in range(b1, min(b1 + pad, maxsize), action):
            broken = False
            for j in range(b2, b3):
                if sobel_y[i, j] > thresh:
                    broken = True
                    if diff == 1:
                        b1 -= 5 * action
                    break
            if broken:
                break
            else:
                b1 += action
                diff = 1
        return (b1, diff)

    def moveboundx(b1, b2, b3, pad, action):
        for j in range(b1, min(b1 + pad, im.shape[1]), action):
            broken = False
            for i in range(b2, b3):
                if sobel_x[i, j] > thresh:
                    broken = True
                    break
            if broken:
                break
            else:
                b1 += action
        return b1

    diff1 = 0
    boundingxy[1], diff1 = moveboundy(
        999999999, boundingxy[1], boundingxy[0], boundingxy[2], -pad, -1, diff1
    )
    diff2 = 0
    boundingxy[3], diff2 = moveboundy(
        gray.shape[0], boundingxy[3], boundingxy[0], boundingxy[2], pad, 1, diff2
    )
    boundingxy[0] = moveboundx(boundingxy[0], boundingxy[1], boundingxy[3], -pad, -1)
    boundingxy[2] = moveboundx(boundingxy[2], boundingxy[1], boundingxy[3], pad, 1)

    if diff1 == 1:
        boundingxy[1] -= 5

    if diff2 == 1:
        boundingxy[3] += 5

    drawing = cv2.rectangle(
        im,
        (boundingxy[0], boundingxy[1]),
        (boundingxy[2], boundingxy[3]),
        (255, 0, 0),
        2,
    )

    res = 50
    cv2.imshow("gray", utils.resize(gray, res))

    if True:
        cv2.imshow("b", utils.resize(im, 50))
        cv2.waitKey(0)

    class Result:
        def __init__(self, boundingxy, drawing):
            self.boundingxy = boundingxy
            self.drawing = drawing

    return Result(boundingxy, drawing)


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
