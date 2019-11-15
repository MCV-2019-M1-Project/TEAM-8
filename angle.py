# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# %load_ext autoreload
# %autoreload 2


# %%
import glob

import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from scipy.ndimage import uniform_filter
from PIL import Image

from utils import show_img

path = "datasets/qsd1_w5"
image_paths = sorted(glob.glob(f"{path}/*.jpg"))

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
    return cv2.HoughLinesP(
        edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )


def draw_lines(img, lines):
    line_image = np.zeros_like(img)  # creating a blank to draw lines on

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    # cv2.addWeighted(img, 0.8, line_image, 1, 0)

    return line_image


def draw_lines_on(img, lines):
    line_img = draw_lines(img, lines)
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)


# %%

def get_all_lines(cvimg, iters=3):
    gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    line_img = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    kernel_size = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in range(iters):
        lines = get_lines(line_img)
        if i != iters - 1:
            line_img = draw_lines(line_img, lines)
            line_img = cv2.dilate(line_img, kernel)

    line_img = draw_lines(line_img, lines)
    # show_img(line_img, title=image_path)
    return lines


def get_angle(line):
    x1, y1, x2, y2 = line
    angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
    weight = euclidean((x1, y1), (x2, y2))
    angle = angle % 90
    return int(angle), weight


def get_horiz_angle(lines):
    weighted_angles = [get_angle(line[0]) for line in lines]
    possible = np.zeros(90)
    for a, w in weighted_angles:
        possible[a] += w
    possible = uniform_filter(possible, size=2)
    final_angle = np.argmax(possible)

    return final_angle


def draw_horizontal_lines(cvimg, lines, horiz_angle):
    def is_close_to_horiz(line):
        return abs(get_angle(line[0])[0] - horiz_angle) < 2

    lines = filter(is_close_to_horiz, lines)

    cvimg = draw_lines_on(cvimg, lines)
    return cvimg


def get_rotation(horiz_angle):
    if horiz_angle > 45:
        rotation = -abs(90 - horiz_angle)
    else:
        rotation = horiz_angle
    return rotation


def get_GTFORMAT_angle(horiz_angle):
    rotation = get_rotation(horiz_angle)
    if rotation <= 0:
        return -rotation
    return 180 - rotation


def get_rotated(path, horiz_angle):
    rotation = get_rotation(horiz_angle)

    img = Image.open(path)
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))
    img = img.rotate(angle=rotation, expand=False)
    return np.array(img)


def read_horizontal_image(path):
    img = cv2.imread(path)
    lines = get_all_lines(img)
    angle = get_horiz_angle(lines)
    return get_rotated(path, angle)


if __name__ == "__main__":
    from utils import get_pickle

    GT = get_pickle("datasets/angles_qsd1w5_v2.pkl")
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        lines = get_all_lines(img)
        angle = get_horiz_angle(lines)
        gt_like_angle = get_GTFORMAT_angle(angle)
        print(f"Detected: {gt_like_angle} Ground Truth: {GT[i]}")
        show_img(draw_horizontal_lines(img, lines, angle))
        show_img(get_rotated(path, angle))
