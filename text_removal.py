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

    blur = cv2.GaussianBlur(im, (15,15), 0)

    sobely64f = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)
    sobely64f = np.abs(sobely64f)
    sobely64f = sobely64f / np.amax(sobely64f)

    sobelx64f = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    sobelx64f = np.abs(sobelx64f)
    sobelx64f = (sobelx64f / np.amax(sobelx64f) * 255).astype("uint8")
    print(sobelx64f.shape)

    h, l, s = cv2.split(cv2.cvtColor(sobelx64f, cv2.COLOR_BGR2HLS))

    grays = cv2.cvtColor(sobelx64f, cv2.COLOR_BGR2GRAY)
    grays_i = grays.astype("float16")
    s_i = s.astype("float16")

    grays_i -= s_i
    grays_i[grays_i < 0] = 0

    grays_i = grays_i / np.amax(grays) * 255
    grays_i = grays_i.astype("uint8")
    grays_i[grays_i < 100] = 0

    kernel = np.ones((15, 15), np.uint8)
    start_x = round(im.shape[1] / 3)
    end_x = round(2 * im.shape[1] / 3)

    min = 0
    min_value = 99999999999999

    im_m = np.copy(im)
    grad_x_neighb = 0
    for y in range(round(im.shape[0] / 4 - 4)):
        j_p = round(3 * im.shape[0] / 4) + y
        grad_devia = 0
        grad_x_neighb = 0

        mean_b = np.mean(blur[j_p][start_x:end_x])
        for x in range(start_x, end_x):
            grad_devia += abs(mean_b - blur[j_p, x])
            #im_m[j_p,x] = [255, 0, 0]

        for x in range(start_x, end_x):
            grad_x_neighb += grays_i[j_p + 1, x]
            grad_x_neighb += grays_i[j_p + 2, x]
            grad_x_neighb += grays_i[j_p + 3, x]

        grad_x_neighb = 0.001 if grad_x_neighb == 0 else grad_x_neighb
        new_value = sum(grad_devia) / grad_x_neighb

        if new_value < min_value:
            print(new_value, " ", grad_x_neighb)
            min_value = new_value
            min = j_p

        ##########

        j_m = round(im.shape[0] / 4) - y
        grad_devia = 0
        grad_x_neighb = 0

        mean_b = np.mean(blur[j_m][start_x:end_x])
        for x in range(start_x, end_x):
            grad_devia += abs(mean_b - blur[j_m, x])
            #im_m[j_m, x] = [0, 255, 00]

        for x in range(start_x, end_x):
            grad_x_neighb += grays_i[j_m - 1, x]
            grad_x_neighb += grays_i[j_m - 2, x]
            grad_x_neighb += grays_i[j_m - 3, x]

        grad_x_neighb = 0.001 if grad_x_neighb == 0 else grad_x_neighb
        new_value = sum(grad_devia) / grad_x_neighb

        if new_value < min_value:
            print(new_value, " ", grad_x_neighb)
            min_value = new_value
            min = j_m

    print("min at ", min_value)

    pad = 3
    im_m[min - pad:min + pad, start_x:end_x] = [0, 0, 255]
    im_m[min, start_x:end_x] = [255, 0, 0]

    cv2.imshow("image", utils.resize(im_m, 25))
    cv2.imshow("sobel", utils.resize(sobelx64f, 25))
    cv2.imshow("saturation", utils.resize(s, 25))
    cv2.imshow("joined", utils.resize(grays_i, 25))

    cv2.waitKey()

    return

    def get_y():
        min = 0
        min_value = 99999999999999

        for y in range(im.shape[0]):
            current_value = 0
            for x in range(start_x, end_x):
                current_value += opening[y, x]

            current_value = sum(current_value) / im.shape[1]
            if current_value < min_value:
                min_value = current_value
                min = y

        return min

    opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    min = get_y()
    pad = 3

    lap = np.array((
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]), dtype="uint8")

    lapa = sci.convolve2d(utils.resize(gray, 25), lap)
    max = np.amax(lapa)
    lap = (lap / max) * 255

    print(np.amax(lap))
    print(np.amin(lap))

    print(abs_sobely64f.shape)

    im[min - pad:min + pad, start_x:end_x] = [0, 0, 255]
    im[min, start_x:end_x] = [255, 0, 0]

    print(min)

    cv2.imshow("b", blur)

    cv2.waitKey()


    # print(im.shape[0])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((22, 22), np.uint8)

    # construct the Sobel x-axis kernel
    sobelX = np.array((
        [0, 0, 0],
        [-1, 0, 1],
        [0, 0, 0]), dtype="int")

    corner = np.array((
        [-1, -0.75, -0.75],
        [-0.75, 1, 1],
        [-0.75, 1, 1]), dtype="uint8")

    opening = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    tophat = cv2.morphologyEx(im, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, kernel)

    h, l, s = cv2.split(cv2.cvtColor(opening, cv2.COLOR_BGR2HLS))
    h2, l2, s2 = cv2.split(cv2.cvtColor(opening, cv2.COLOR_BGR2HLS))

    cornered = sci.convolve2d(utils.resize(gray, 25), corner)
    maxValue = np.amax(cornered)
    cornered[cornered < maxValue] = 0
    print(maxValue)

    opening = cv2.morphologyEx(s2, cv2.MORPH_OPEN, kernel)
    #opening = cv2.morphologyEx(s2, cv2.MORPH_CLOSE, kernel)

    blur = cv2.GaussianBlur(opening, (19, 19), 0)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)


    return

    boundingxy = [
        boundRect[-1][0] + 5,
        boundRect[-1][1] + 5,
        boundRect[-1][0] + boundRect[-1][2] - 5,
        boundRect[-1][1] + boundRect[-1][3] - 5,
    ]

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    imager = im[:, :, 0]
    imageg = im[:, :, 1]
    imageb = im[:, :, 2]
    output = np.equal(imager, imageg, dtype=int)
    output2 = np.equal(imager, imageb, dtype=int)
    output1 = np.equal(imageb, imageg, dtype=int)
    output3 = output & output2
    output4 = output3 & output1
    mask = 255 * output4
    mask1 = mask.astype(np.uint8)
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(
    # gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((int((im.shape[0]) / 350), int((im.shape[1]) / 35)), np.uint8)

    denoised = cv2.erode(mask1, kernel, iterations=5)
    # denoised = cv2.erode(denoised, kernel, iterations=1)
    denoised = cv2.dilate(denoised, kernel, iterations=3)
    denoised = cv2.dilate(denoised, kernel, iterations=2)

    canny_output = cv2.Canny(denoised, 200, 255)
    contours, hierarchy = cv2.findContours(
        canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    drawing = np.zeros(
        (canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8
    )
    drawing[:, :, 0] = gray
    drawing[:, :, 1] = gray
    drawing[:, :, 2] = gray
    max_area = 0
    max = 0
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        area = boundRect[i][2] * boundRect[i][3]
        if (area > max_area) & (boundRect[i][3] < boundRect[i][2]):
            max = boundRect[i]
            max_area = area
        # FIXME(Beto): If there're two areas of the same size Eddie says it's not worki
        if area == max_area:
            max = boundRect[i]
        boundRect[i] = max

    for i in range(len(contours)):
        # cv2.drawContours(drawing, contours_poly, i, (0, 255, 0))
        cv2.rectangle(
            drawing,
            (int(boundRect[-1][0]), int(boundRect[-1][1])),
            (
                int(boundRect[-1][0] + boundRect[-1][2]),
                int(boundRect[-1][1] + boundRect[-1][3]),
            ),
            (0, 255, 0),
            2,
        )

    boundingxy = [
        boundRect[-1][0] + 5,
        boundRect[-1][1] + 5,
        boundRect[-1][0] + boundRect[-1][2] - 5,
        boundRect[-1][1] + boundRect[-1][3] - 5,
    ]

    # Betos Post-Processing from here on

    # Otsu's thresholding
    imx = np.copy(gray[boundingxy[1] : boundingxy[3], boundingxy[0] : boundingxy[2]])
    ima, th2 = cv2.threshold(imx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (maxkernel + 6, maxkernel + 6))

    if diffmin < diffmax:
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    else:
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    grad_x = cv2.Sobel(
        gray,
        ddepth,
        1,
        0,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT,
    )
    grad_y = cv2.Sobel(
        gray,
        ddepth,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT,
    )

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    thresh = 15

    grad[boundingxy[1] : boundingxy[3], boundingxy[0] : boundingxy[2]] = 0

    def moveboundy(maxsize, b1, b2, b3, pad, action, diff=-1):
        for i in range(b1, min(b1 + pad, maxsize), action):
            broken = False
            for j in range(b2, b3):
                if grad[i, j] > thresh:
                    broken = True
                    if diff == 1:
                        b1 -= 2 * action
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
                if grad[i, j] > thresh:
                    broken = True
                    break
            if broken:
                break
            else:
                b1 += action
        return b1

    diff1 = 0
    boundingxy[1], diff1 = moveboundy(
        999999999, boundingxy[1], boundingxy[0], boundingxy[2], -50, -1, diff1
    )
    diff2 = 0
    boundingxy[3], diff2 = moveboundy(
        grad.shape[0], boundingxy[3], boundingxy[0], boundingxy[2], 50, 1, diff2
    )
    boundingxy[0] = moveboundx(boundingxy[0], boundingxy[1], boundingxy[3], -50, -1)
    boundingxy[2] = moveboundx(boundingxy[2], boundingxy[1], boundingxy[3], 50, 1)

    drawing = cv2.rectangle(
        drawing,
        (boundingxy[0], boundingxy[1]),
        (boundingxy[2], boundingxy[3]),
        (0, 255, 255),
        2,
    )

    if diff1 == 1:
        boundingxy[1] -= 2

    if diff2 == 1:
        boundingxy[3] += 2

    if True:
        cv2.imshow("b", utils.resize(im, 50))
        cv2.imshow("c", utils.resize(drawing, 50))
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
