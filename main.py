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
    return [cv.imread(path) for path in tqdm(paths)]


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
    if des1 is None:
        return

    if des2 is None:
        return

    if method == "BF":
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Sort matches in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[0: min(10, len(matches))]
        return matches


def evaluate_pair(matches):
    result = 0

    if matches is None:
        return 999999

    for match in matches:
        result += match.distance
    return result / len(matches)


def main():
    # Get images and denoise query set.
    print("Getting images...")
    qs = get_imgs("datasets/qsd1_w3")
    db = get_imgs("datasets/DDBB")
    qs_denoised = [denoise_imgs(img) for img in tqdm(qs)]

    # Get mask without background and without text box of query sets.
    qs_bck_masks = [get_mask_background(img) for img in tqdm(qs_denoised)]
    qs_bb_masks = [get_mask_text(img) for img in tqdm(qs_denoised)]

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

    print("Detecting and describing keypoints...")
    qs_kps = [detect_keypoints(dt_type, img, mask) for img, mask in tqdm(zip(qs_denoised[0:2], qs_masks[0:2]))]
    qs_dps = [describe_keypoints(dt_type, img, kp) for img, kp in tqdm(zip(qs_denoised[0:2], qs_kps[0:2]))]

    db_kps = [detect_keypoints(dt_type, img) for img in tqdm(db)]
    db_dps = [describe_keypoints(dt_type, img, kp) for img, kp in tqdm(zip(db, db_kps))]

    # Match images
    print("Matching images...")

    matches = [match_descriptions(qs_dps[0], db_dp) for db_dp in tqdm(db_dps)]
    matches_ev = [evaluate_pair(match) for match in matches]

    class Match:
        def __init__(self, strength, idx):
            self.strength = strength
            self.idx = idx

    matches_pck = [Match(strength, idx) for strength, idx in zip(matches_ev, range(len(matches_ev)))]
    matches_pck = sorted(matches_pck, key=lambda x: x.strength)

    print([match_pck.idx for match_pck in matches_pck])
    print([match_pck.strength for match_pck in matches_pck])




    matches = [match_descriptions(qs_dps[1], db_dp) for db_dp in tqdm(db_dps)]
    matches_ev = [evaluate_pair(match) for match in matches]

    matches_pck = [Match(strength, idx) for strength, idx in zip(matches_ev, range(len(matches_ev)))]
    matches_pck = sorted(matches_pck, key=lambda x: x.strength)

    print([match_pck.idx for match_pck in matches_pck])
    print([match_pck.strength for match_pck in matches_pck])
    print([pair.distance for pair in matches[1]])


    if SHOW_IMGS:
        img_matches = 0
        img_matches = cv.drawMatches(qs_denoised[1], qs_kps[1], db[matches_pck[1].idx], db_kps[matches_pck[1].idx], matches[1], img_matches)
        cv.imshow("a", qs_denoised[0])
        cv.imshow("b", qs_denoised[1])
        cv.imshow("matches", img_matches)
        cv.waitKey()

    print("END")


main()
