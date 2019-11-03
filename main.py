import pickle
import numpy as np
import cv2 as cv
import fire
import ml_metrics as metrics
from tqdm.auto import tqdm
import glob

import utils
import text_removal
import background_remover

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
#                                       Compare additionally with read text
#                                       SIFT (PATENTED, requires an older version of openCV)
#                                       SURF (PATENTED, requires an older version of openCV)
#                                       HOG
#                                       ...

# TASK 2:       Find tentative matches based on similarity of local appearance and verify matches.
#                       Beto implemented brute force based matching.
#                       Beto implemented flann based matching.
#               TODO Changing tops list to [-1] if image is not present in dataset

# TASK 3:       Evaluate system on QSD1-W4, map@k.

# TASK 4:       TODO Evaluate best system from W3 on QSD1-W4.


SHOW_IMGS = True


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


def get_text_bb_info(img):
    return text_removal.getpoints2(img)


def detect_keypoints(detection_type, img, mask=None):
    kp = detection_type.detect(img, mask)
    return kp


def describe_keypoints(description_type, img, kp):
    kp, des = description_type.compute(img, kp)
    return des


def match_descriptions(des1, des2, method="BRUTE_FORCE", lowe_filter=False):
    if des1 is None or des2 is None:
        return

    matches = []
    matches_unfiltered =[]

    if method == "BRUTE_FORCE":
        if lowe_filter:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
            matches_unfiltered = bf.knnMatch(des1, des2, k=2)
        else:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

    elif method == "FLANN":
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=100)

        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches_unfiltered = flann.knnMatch(des1, des2, k=2)
        lowe_filter = True

    if lowe_filter:
        for i, pair in enumerate(matches_unfiltered):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.7 * n.distance:
                matches.append(m)

    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[0: min(len(matches), 10)]
    return matches


def evaluate_matches(matches):
    if matches is None or len(matches) == 0:
        return 999999

    result = 0
    for match in matches:
        result += match.distance
    return result / len(matches)


def comparing_with_ground_truth(tops, txt_infos, k):
    utils.dump_pickle("result.pkl", tops)
    gt = utils.get_pickle("datasets/qsd1_w4/gt_corresps.pkl")
    hypo = utils.get_pickle("result.pkl")
    mapAtK = metrics.mapk(gt, hypo, k)
    print("\nMap@ " + str(k) + " is " + str(mapAtK))

    bbs_gt = np.asarray(utils.get_groundtruth("datasets/qsd1_w4/text_boxes.pkl")).squeeze()
    bbs_predicted = [txt_info.boundingxy for txt_info in txt_infos]
    mean_iou = utils.get_mean_IoU(bbs_gt, bbs_predicted)
    print("Mean Intersection over Union: ", mean_iou)

    texts_gt = utils.get_gt_text("datasets/qsd1_w4")
    texts_predicted = [txt_info.text for txt_info in txt_infos]
    mean_lev = utils.compute_lev(texts_gt, texts_predicted)
    print(texts_predicted)
    print("\n")
    print(texts_gt)
    print("Mean Levenshtein distance: ", mean_lev)


def main():
    #K parameter for map@k
    k = 10
    # Get images and denoise query set.
    print("Getting and denoising images...")
    qs = get_imgs("datasets/qsd1_w4")
    db = get_imgs("datasets/DDBB")
    qs_denoised = [denoise_imgs(img) for img in tqdm(qs)]

    #Separating paitings inside images to separate images
    qs_split = [background_remover.remove_background(img) for img in qs_denoised]
    # Get masks without background and without text box of query sets.
    print("\nGetting text bounding box masks...")
    #Not needed since the above function already crops the background
    #qs_bck_masks = [get_mask_background(img) for img in tqdm(qs_denoised)]
    qs_txt_infos = [[get_text_bb_info(painting) for painting in img] for img in tqdm(qs_split)]
    qs_txt_masks = [[single.mask for single in qs_txt_info] for qs_txt_info in qs_txt_infos]

    for qs_mask in qs_txt_masks:
        for single_mask in qs_mask:
            single_mask[single_mask < 255] = 0
            single_mask[single_mask > 255] = 255

    qs_masks = [[single_mask.astype("uint8") for single_mask in qs_mask] for qs_mask in qs_txt_masks]

    # Detect and describe keypoints in images.
    print("\nDetecting and describing keypoints...")
    dt_type = cv.ORB_create()
    qs_kps = [[detect_keypoints(dt_type, painting, painting_mask) for painting, painting_mask in zip(img, mask)]
              for img, mask in zip(qs_split, qs_masks)]
    qs_dps = [[describe_keypoints(dt_type, painting, painting_kp) for painting, painting_kp in zip(img, kp)]
              for img, kp in zip(qs_split, qs_kps)]

    db_kps = [detect_keypoints(dt_type, img) for img in tqdm(db)]
    db_dps = [describe_keypoints(dt_type, img, kp) for img, kp in tqdm(zip(db, db_kps))]

    # Match images
    print("\nMatching images...")

    class Match:
        def __init__(self, summed_dist, idx):
            self.summed_dist = summed_dist
            self.idx = idx

    tops = []
    dists = []

    # For all query images
    for qs_dp in tqdm(qs_dps):
        # Get all descriptor matches between a query image and all database images.
        matches_s = [[match_descriptions(qs_single_painting_dp, db_dp) for qs_single_painting_dp in qs_dp] for db_dp in db_dps]
        # Evaluate quality of matches
        matches_s_ev = [[evaluate_matches(painting_match) for painting_match in match] for match in matches_s]
        # Sort for lowest
        matches_s_cl = [[Match(painting_summed_dist, idx) for painting_summed_dist in summed_dist] for idx, summed_dist in enumerate(matches_s_ev)]
        if len(qs_dp) > 1:
            p1 = [match[0] for match in matches_s_cl]
            p2 = [match[1] for match in matches_s_cl]
            p1 = sorted(p1, key=lambda x: x.summed_dist)
            p2 = sorted(p2, key=lambda x: x.summed_dist)
            sorted_list = [p1, p2]
        else:
            p1 = [match[0] for match in matches_s_cl]
            p1 = sorted(p1, key=lambda x: x.summed_dist)
            sorted_list = [p1]
        tops.append([[matches.idx for matches in painting[0:k]] for painting in sorted_list])
        dists.append([[matches.summed_dist for matches in painting[0:k]] for painting in sorted_list])

    #Removing results with too big of a distance
    for i, im in enumerate(tops):
        for j, painting in enumerate(im):
            #Distance threshold
            if dists[i][j][0] > 35:
                tops[i][j] = [-1]

    comparing_with_ground_truth(tops, qs_txt_infos, k)

    if SHOW_IMGS:
        img_matches = 0
        img_matches = cv.drawMatches(qs_denoised[1], qs_kps[1], db[matches_s_cl[1].idx], db_kps[matches_s_cl[1].idx], matches_s[1], img_matches)
        rezised = cv.resize(img_matches,(int(img_matches.shape[1] * 50/100),int(img_matches.shape[0] * 50/100)))
        # cv.imshow("a", qs_denoised[0])
        # cv.imshow("b", qs_denoised[1])
        cv.imshow("matches", rezised)
        cv.waitKey()


main()
