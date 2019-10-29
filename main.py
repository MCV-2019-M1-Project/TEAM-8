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

# PRE-TASK -3:  Remove Noise.
# PRE-TASK -2:  Split images. #TODO
# PRE-TASK -1:  Mask background. #TODO
# PRE-TASK 0:   Mask text bounding box.

# TASK 1:       Detect keypoints and compute descriptors.
#                       Step 1: Detection
#                               Beto implemented ORB.
#                       Step 2: Description
#                               Beto implemented ORB.

# TASK 2:       Find tentative matches based on similarity of local appearance and verify matches.
#                       Beto implemented brute force matching.

# TASK 3:       Evaluate system on QSD1-W4, map@k. #TODO

# TASK 4:       Evaluate best system from W3 on QSD1-W4. #TODO

SHOW_IMGS = True


def get_imgs(files_path, extension ="jpg"):
    paths = sorted(glob.glob(f"{files_path}/*." + extension))
    return [cv.imread(path) for path in paths]


def denoise_imgs(img):
    return cv.medianBlur(img, 3)


def split_imgs(img):
    return [img]


def get_mask_background(img): # TODO
    mask = np.full((img.shape[0], img.shape[1]), 255, dtype="uint16")
    return mask


def get_mask_text(img):
    return text_removal.getpoints2(img).mask


def detect_keypoints(detection_type, img, mask=None):
    kp = detection_type.detect(img, mask)
    return kp


def describe_keypoints(description_type, img, kp):
    kp, des = description_type.compute(img, kp)
    return des


def match_descriptions(des1, des2, method="BF"):
    if method == "BF":
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Sort matches in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        return matches


def evaluate_pair(matches):
    result = 0
    for match in matches:
        result += match.distance
    return result


def main():
    # Get images and denoise query set.
    qs = get_imgs("datasets/qsd1_w3")
    db = cv.imread("datasets/DDBB/bbdd_00131.jpg")
    qs_denoised = [denoise_imgs(img) for img in qs[6:7]]

    # Get mask without background and without text box of query sets.
    qs_bck_masks = [get_mask_background(img) for img in qs_denoised]
    qs_bb_masks = [get_mask_text(img) for img in qs_denoised]

    # Merge masks into a single mask
    qs_masks = [
        bck_mask.astype("uint16") + bb_mask.astype("uint16")
        for bck_mask, bb_mask in zip(qs_bck_masks, qs_bb_masks)]

    for qs_mask in qs_masks:
        qs_mask[qs_mask <= 255] = 0
        qs_mask[qs_mask > 255] = 255

    qs_masks = [qs_mask.astype("uint8") for qs_mask in qs_masks]

    # Detect and describe keypoints in images.
    dt_type = cv.ORB_create()

    qs_kp = detect_keypoints(dt_type, qs_denoised[0], qs_masks[0])
    qs_dp = describe_keypoints(dt_type, qs_denoised[0], qs_kp)

    db_kp = detect_keypoints(dt_type, db)
    db_dp = describe_keypoints(dt_type, db, db_kp)

    # Match images
    matches = match_descriptions(qs_dp, db_dp)

    if SHOW_IMGS:
        img_matches = 0
        img_matches = cv.drawMatches(qs_denoised[0], qs_kp, db, db_kp, matches[:10], img_matches)
        cv.imshow("matches", img_matches)
        cv.waitKey()

    print("END")


main()
