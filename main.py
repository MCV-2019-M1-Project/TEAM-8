import numpy as np
import cv2 as cv
import ml_metrics as metrics
from tqdm.auto import tqdm
import glob

import utils
import text_removal
import angle as ag
import background_remover

SHOW_IMGS = False

def get_imgs(files_path, extension="jpg"):
    paths = sorted(glob.glob(f"{files_path}/*." + extension))
    return [cv.imread(path) for path in tqdm(paths)]


# TODO: Check if best option
def denoise_imgs(img):
    return cv.medianBlur(img, 3)


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
    texts_predicted = [[painting.text for painting in txt_info] for txt_info in txt_infos]
    for i, item in enumerate(texts_predicted):
        with open('outputs/' + f'{i:05}' + '.txt', 'w') as f:
            for text in item:
                f.write("%s\n" % text)

    gt = utils.get_pickle("datasets/qsd1_w5/gt_corresps.pkl")
    mapAtK = utils.compute_mapk(gt, tops, k)
    print("\nMap@ " + str(k) + " is " + str(mapAtK))

    bbs_gt = np.asarray(utils.get_groundtruth("datasets/qsd1_w5/text_boxes.pkl")).squeeze()
    bbs_predicted = [[painting.boundingxy for painting in txt_info] for txt_info in txt_infos]
    mean_iou = utils.get_mean_IoU(bbs_gt, bbs_predicted)
    print("Mean Intersection over Union: ", mean_iou)

    texts_gt = utils.get_gt_text("datasets/qsd1_w5")
    texts_predicted = [[painting.text for painting in txt_info] for txt_info in txt_infos]
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
    qs = get_imgs("datasets/qst1_w5")
    db = get_imgs("datasets/DDBB")
    #gt_boxes = utils.get_pickle("datasets/qsd1_w5/frames.pkl")
    qs_denoised = [utils.denoise_image(img, "Median") for img in tqdm(qs)]

    print("Generating background masks")
    bg_masks = [utils.get_painting_mask(img, 0.1) for img in tqdm(qs)]
    frame_rectangles = [utils.get_frames_from_mask(mask) for mask in bg_masks]
    #Stan's method
    img_lines = [ag.get_all_lines(img) for img in qs]
    angles = [ag.get_horiz_angle(lines) for lines in img_lines]
    corrected_angles = [ag.get_GTFORMAT_angle(single_angle) for single_angle in angles]
    #Marc's method
    angles_opencv = [utils.get_median_angle(image_rects) for image_rects in frame_rectangles]
    boxes = [[utils.get_box(rectangle) for rectangle in image] for image in frame_rectangles]
    boxes_result = [[[angle, box] for box in image] for angle, image in zip(corrected_angles, boxes)]
    print("Recovering subimages")
    qs_split = [utils.get_paintings_from_frames(img, rects) for img, rects in tqdm(zip(qs_denoised, frame_rectangles))]

    if SHOW_IMGS:
        for i, img in enumerate(tqdm(qs_split)):
            for j, painting in enumerate(img):
                #s = cv.imwrite(r"outputs\0%d%d.jpg"%(i,j), painting)
                cv.imshow("I: " + str(i) + " P: " + str(j), cv.resize(painting, (256, 256)))
                cv.waitKey(0)
                #print(s)

    # Get masks without background and without text box of query sets.
    print("\nGetting text bounding box masks...")
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
    dst_thr = 35
    for qs_dp in tqdm(qs_dps):
        # Get all descriptor matches between a query image and all database images.
        matches_s = [[match_descriptions(qs_single_painting_dp, db_dp) for qs_single_painting_dp in qs_dp] for db_dp in db_dps]
        # Evaluate quality of matches
        matches_s_ev = [[evaluate_matches(painting_match) for painting_match in match] for match in matches_s]
        # Sort for lowest
        matches_s_cl = [[Match(painting_summed_dist, idx) for painting_summed_dist in summed_dist] for idx, summed_dist in enumerate(matches_s_ev)]
        partial_tops, partial_dists = utils.get_tops_from_matches(qs_dp, matches_s_cl, dst_thr, k)
        tops.append(partial_tops)
        dists.append(partial_dists)

    utils.dump_pickle("outputs/frames.pkl", boxes_result)
    utils.dump_pickle("outputs/result.pkl", tops)
    exit()
    comparing_with_ground_truth(tops, qs_txt_infos, k)


main()

