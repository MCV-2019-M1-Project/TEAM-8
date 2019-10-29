"""
Example usage: python run.py task2
"""
import pickle
import numpy as np
import cv2 as cv
import fire
import ml_metrics as metrics
from tqdm.auto import tqdm
import utils
import text_removal

import glob

# TASK 1: Detect keypoints and compute descriptors
#   Step 1: Detection
#   Step 2: Description

# TASK 2: Find tentative matches based on similarity of local appearance and verify matches

# TASK 3: Evaluate system on QSD1-W4, map@k

# TASK 4: Evaluate best system from W3 on QSD1-W4


def get_images(files_path, extension = "jpg"):
    paths = sorted(glob.glob(f"{files_path}/*." + extension))
    return [cv.imread(path) for path in paths]


def denoise(img):
    return cv.medianBlur(img, 3)


def detect_keypoints(detection_type, img, mask=None):
    kp = detection_type.detect(img, mask)
    return kp


def describe_keypoints(description_type, img, kp):
    kp, des = description_type.compute(img, kp)
    return kp, des


def match_descriptions(des1, des2, method="BF"):
    if method == "BF":
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Sort matches in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        return matches


def main():
    # Get images and denoise query set.
    qs = get_images("datasets/qsd1_w3")
    db = cv.imread("datasets/DDBB/bbdd_00131.jpg")
    qs_denoised = [denoise(img) for img in qs[6:7]]

    # TODO: Add image separation and background removal

    # Get mask without text box of query sets.
    qs_bb_infos = [text_removal.getpoints2(img) for img in qs_denoised]
    qs_bb_masks = [bb_info.mask for bb_info in qs_bb_infos]

    # Detect and describe keypoints in images.
    dt_type = cv.ORB_create()

    qs_kp = detect_keypoints(dt_type, qs_denoised[0], qs_bb_masks[0])
    qs_kp, qs_des = describe_keypoints(dt_type, qs_denoised[0], qs_kp)

    db_kp = detect_keypoints(dt_type, db)
    db_kp, db_des = describe_keypoints(dt_type, qs_denoised[0], db_kp)

    # Match images
    match = match_descriptions(qs_des, db_des)

    img3 = 0
    img3 = cv.drawMatches(qs_denoised[0], qs_kp, db, db_kp, match[:10], img3)
    cv.imshow("matches", img3)
    cv.waitKey()

    print("END")

main()