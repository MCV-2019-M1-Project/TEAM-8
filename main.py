"""
Example usage: python run.py task2
"""
import pickle
import numpy as np
import cv2 as cv
import fire
import ml_metrics as metrics
from tqdm.auto import tqdm
import glob

import utils
import text_removal

# PRE-TASK -3:  Remove Noise.
# PRE-TASK -2:  TODO Split images.
# PRE-TASK -1:  TODO Mask background.
# PRE-TASK 0:   Mask text bounding box.

# TASK 1:       Detect keypoints and compute descriptors.
#                       Step 1: Detection
#                               Beto implemented ORB.
#                               More possible variations:
#                                       Harris Laplacian
#                                       Difference of Gaussians (DoG or SIFT)
#                                       ...
#                       Step 2: Description
#                               Beto implemented ORB.
#                               More possible variations:
#                                       SIFT
#                                       SURF
#                                       HOG
#                                       ...

# TASK 2:       Find tentative matches based on similarity of local appearance and verify matches.
#                       Beto implemented brute force matching.
#                       More possible variations:
#                               https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
#                                       FLANN based matching
#                                       Add Lowe Filter to BF or FLANN

# TASK 3:       TODO Evaluate system on QSD1-W4, map@k.

# TASK 4:       TODO Evaluate best system from W3 on QSD1-W4.


SHOW_IMGS = False


def get_imgs(files_path, extension ="jpg"):
    paths = sorted(glob.glob(f"{files_path}/*." + extension))
    return [cv.imread(path) for path in tqdm(paths)]


# TODO: Check if best option
def denoise_imgs(img):
    return cv.medianBlur(img, 3)


# TODO
def split_imgs(img):
    return [img]


# TODO
def get_mask_background(img):
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


def evaluate_matches(matches):
    result = 0

    if matches is None:
        return 999999

    for match in matches:
        result += match.distance
    return result / len(matches)


def main():
    # Get images and denoise query set.
    print("Getting and denoising images...")
    qs = get_imgs("datasets/qsd1_w3")
    db = get_imgs("datasets/DDBB")
    qs_denoised = [denoise_imgs(img) for img in tqdm(qs)]

    # Get mask without background and without text box of query sets.
    print("\nGetting background and text bounding box masks...")
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
    print("\nDetecting and describing keypoints...")
    dt_type = cv.ORB_create()

    qs_kps = [detect_keypoints(dt_type, img, mask) for img, mask in zip(qs_denoised, qs_masks)]
    qs_dps = [describe_keypoints(dt_type, img, kp) for img, kp in zip(qs_denoised, qs_kps)]

    db_kps = [detect_keypoints(dt_type, img) for img in tqdm(db)]
    db_dps = [describe_keypoints(dt_type, img, kp) for img, kp in tqdm(zip(db, db_kps))]

    # Match images
    print("\nMatching images...")

    class Match:
        def __init__(self, summed_dist, idx):
            self.summed_dist = summed_dist
            self.idx = idx

    tops = []

    # For all query images
    for qs_dp in tqdm(qs_dps):
        # Get all descriptor matches between a query image and all database images.
        matches_s = [match_descriptions(qs_dp, db_dp) for db_dp in db_dps]
        # Evaluate quality of matches
        matches_s_ev = [evaluate_matches(match) for match in matches_s]
        # Sort for lowest
        matches_s_cl = [Match(summed_dist, idx) for summed_dist, idx in zip(matches_s_ev, range(len(matches_s_ev)))]
        matches_s_cl = sorted(matches_s_cl, key=lambda x: x.summed_dist)
        tops.append([matches.idx for matches in matches_s_cl[0:10]])

    gt = utils.get_pickle("datasets/qsd1_w3/gt_corresps.pkl")
    k = 10
    mapAtK = metrics.mapk(gt, tops, 10)
    print("Map@" + str(k) + "is" + str(mapAtK))

    if SHOW_IMGS:
        img_matches = 0
        img_matches = cv.drawMatches(qs_denoised[1], qs_kps[1], db[matches_s_cl[1].idx], db_kps[matches_s_cl[1].idx], matches_s[1], img_matches)
        cv.imshow("a", qs_denoised[0])
        cv.imshow("b", qs_denoised[1])
        cv.imshow("matches", img_matches)
        cv.waitKey()


main()
