# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %load_ext autoreload
# %autoreload 2


# %%
from itertools import chain
import glob

import numpy as np
from scipy.fftpack import dct
import cv2

from segmentation import get_last
from utils import show_img


# %%

def get_lines(img):
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(img, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 200  # minimum number of pixels making up a line
    max_line_gap = 50  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    return cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)


def draw_lines(img, lines)
    line_image = np.zeros_like(img)  # creating a blank to draw lines on

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

    return line_image


# %%

path = "datasets/qsd1_w5"
image_paths = sorted(glob.glob(f"{path}/*.jpg"))


def one(image_path):
    img = get_last(image_path)
    show_img(img)
    kernel_size = 7
    blur_gray = cv2.medianBlur(img,kernel_size)
    show_img(img)
    line_img = get_lines(blur_gray)
    show_img(line_img, title=image_path)


def two(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    line_img = get_lines(blur_gray)
    show_img(line_img, title=image_path)


def three(image_path):

    img = cv2.imread(image_path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    img = np.zeros(img.shape, np.uint8)
    print(f"SHAPE {img.shape}")
    img = cv2.drawContours(img, contours, -1, (255,255,255), 3)
    show_img(img)
    line_img = get_lines(img)
    show_img(line_img, title=image_path)


def four(image_path, iters=3):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    show_img(gray)

    kernel_size = 5
    line_img = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    kernel_size = 40    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    for i in range(iters):
        line_img = get_lines(line_img)
        # show_img(line_img, title=image_path)
        if i != iters - 1:
            line_img = cv2.dilate(line_img, kernel)
            # show_img(line_img, title=image_path)
    show_img(line_img, title=image_path)



for path in image_paths:
    print(path)
    # one(path)
    # two(path)
    # three(path)
    four(path)

