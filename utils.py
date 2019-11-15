import pickle

import cv2
import numpy as np
from ml_metrics import apk
from tqdm.auto import tqdm
import math

from mask_metrics import MaskMetrics
import distance as dist
import Levenshtein as lev
import glob
from skimage import feature

import background_remover


def calc_similarities(measure, db, qs, show_progress=False):
    """
    Returns an array of size (qs_size x db_size)
    where arr[i,j] = similarity between
    i-th image in queryset and j-th image in database
    """

    def compute_one(hist):
        result = [measure(hist, db_hist) for db_hist in db]
        return result

    generator = tqdm(qs) if show_progress else qs

    return np.array([compute_one(hist) for hist in generator])


def calc_multi_similarities(measure, db, qs, show_progress=False):
    def compute_one(hists):
        result = np.array([measure(hist, db_hist) for hist in hists for db_hist in db])
        return result

    generator = tqdm(qs) if show_progress else qs
    return [compute_one(hist) for hist in generator]


def normalize_hist(hs):

    h_max = max(max(hs[0]), max(hs[1]), max(hs[2]))
    hs = np.true_divide(hs, h_max)

    return hs


def get_tops(similarities, k):
    """
    Returns an array of size (qs_size x k)
    where arr[i,j] is the index of j-th closest image in the database
    to i-th image in the queryset
    """
    tops = similarities.argsort(axis=1)[:, :k].tolist()
    return tops


def get_multi_tops(similarities, k, dbsize):
    tops = np.array([np.argsort(sim)[:k] for sim in similarities])
    return tops % dbsize


def get_groundtruth(path):
    """
    Returns a list of lists from a specified pickle file
    with the format needed to execute Map@k
    list[[i]] contains the correct prediction for the i-th image
    in the queryset
    """
    pklFile = open(path, "rb")
    groundTruth = pickle.load(pklFile)

    return [[item[0]] for item in groundTruth]


def get_mask_metrics(pred, gt, show_progress):
    results = np.zeros(3)
    generator = tqdm(gt) if show_progress else gt
    for i, gt_mask in enumerate(generator):
        metrics = MaskMetrics(pred[i], gt_mask)
        results[0] += metrics.precision()
        results[1] += metrics.recall()
        results[2] += metrics.f1_score()
    results /= len(gt)
    metrics_dict = dict()
    metrics_dict["precision"] = results[0]
    metrics_dict["recall"] = results[1]
    metrics_dict["f1_score"] = results[2]
    return metrics_dict


def get_mean_IoU(gts, preds):
    result = 0
    for x in range(len(gts)):
        res = dist.intersection_over_union(gts[x], preds[x])
        if res < 0.8:
            print("  Mean IoU at:", x, ":", dist.intersection_over_union(gts[x], preds[x]))
        result += res
    return result / len(preds)


def getgradient(img):
    x, y = np.gradient(img)
    return np.hypot(x, y).astype(np.uint8)


def lapl_at_index(source, index):
    i, j = index
    val = (
        (4 * source[i, j])
        - (1 * source[i + 1, j])
        - (1 * source[i - 1, j])
        - (1 * source[i, j + 1])
        - (1 * source[i, j - 1])
    )
    return val


def resize(img, percent):
    scale_percent = percent  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def binsearch(p, q, cond):
    mid = (p + q) // 2
    if mid == p:
        return mid
    elif cond(mid):
        return binsearch(mid, q, cond)
    return binsearch(p, mid, cond)


def show_img(img, title=""):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def compute_lev(gts, preds):
    result = 0
    preds_string = [''.join(map(str, item))for item in preds]
    for x in range(len(preds)):
        res = lev.distance(gts[x], preds_string[x])
        if res != 0:
            print("  Lev distance at", x, ":", res)
        result += res
    return result / len(preds)


def get_gt_text(path):
    paths = sorted(glob.glob(f"{path}/*.txt"))
    result = []
    for path in paths:
        result.append(open(path, "r").read())
        # result.append(open(path, "r").read().split("'")[1])
    return result


def get_hog_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    descriptors = feature.hog(img, orientations=9, pixels_per_cell=(8, 8))
    return descriptors.ravel()


def get_hog_histograms(imgs):
    hogs_imgs = [get_hog_histogram(imgs[x]) for x in tqdm(range(len(imgs)))]
    return hogs_imgs

def denoise_image(img, method="Gaussian"):
    if method == "Gaussian":
        return cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    elif method == "Median":
        return cv2.medianBlur(img, 3)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif method == "FastNl":
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def get_hog_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    descriptors = feature.hog(img, orientations=9, pixels_per_cell=(8, 8))
    return descriptors.ravel()


def get_hog_histograms(imgs):
    hogs_imgs = [get_hog_histogram(imgs[x]) for x in tqdm(range(len(imgs)))]
    return hogs_imgs


def list_argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def split_image_recursive(img):
    subimage_list = background_remover.split_image(img)
    if len(subimage_list) == 1:
        return subimage_list
    else:
        # We guess single image on the left
        left = [subimage_list[0]]
        for painting in background_remover.split_image(subimage_list[1]):
            left.append(painting)
        # We guess single image on the right
        right = []
        for painting in background_remover.split_image(subimage_list[0]):
            right.append(painting)
        right.append(subimage_list[1])
        # We take the result with most paintings
        return left if len(left) > len(right) else right


def get_tops_from_matches(qs_dp, matches_s_cl, dst_thr, k):
    if len(qs_dp) > 1:
        p1 = [match[0] for match in matches_s_cl]
        p2 = [match[1] for match in matches_s_cl]
        p1 = sorted(p1, key=lambda x: x.summed_dist)
        p2 = sorted(p2, key=lambda x: x.summed_dist)
        p1_tops = [matches.idx for matches in p1[0:k]]
        p1_dists = [matches.summed_dist for matches in p1[0:k]]
        p2_tops = [matches.idx for matches in p2[0:k]]
        p2_dists = [matches.summed_dist for matches in p2[0:k]]
        if p1_dists[0] > dst_thr:
            p1_tops = [-1]
        if p2_dists[0] > dst_thr:
            p2_tops = [-1]
        return [p1_tops, p2_tops], [p1_dists, p2_dists]
    else:
        p1 = [match[0] for match in matches_s_cl]
        p1 = sorted(p1, key=lambda x: x.summed_dist)
        p1_tops = [matches.idx for matches in p1[0:k]]
        p1_dists = [matches.summed_dist for matches in p1[0:k]]
        if p1_dists[0] > dst_thr:
            p1_tops = [-1]
        return [p1_tops], p1_dists


list_depth = lambda L: isinstance(L, list) and max(map(list_depth, L))+1


def add_list_level(input_list):
    out = []
    for ll in input_list:
        tmp = []
        for q in ll:
            tmp.append([q])
        out.append(tmp)
    return (out)


def compute_mapk(gt, hypo, k_val):

    hypo = list(hypo)
    if list_depth(hypo) == 2:
        hypo = add_list_level(hypo.copy())

    apk_list = []
    for ii, query in enumerate(gt):
        for jj, sq in enumerate(query):
            apk_val = 0.0
            if len(hypo[ii]) > jj:
                apk_val = apk([sq], hypo[ii][jj], k_val)
            apk_list.append(apk_val)

    return np.mean(apk_list)


def get_painting_mask(img, area_thr=0.08):
    #Blur to get rid of soft edges
    blurred = cv2.medianBlur(cv2.medianBlur(img, 9), 9)
    #Get edges with canny and dilate
    ratio = 4
    canny = cv2.Canny(blurred, 12, 12*ratio)
    canny = cv2.dilate(canny, None, iterations=5)
    #Get contours and area of contours in canny image
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_with_area = [[contour, cv2.contourArea(contour)] for contour in contours]
    sorted_contours = sorted(contours_with_area, key=lambda x: x[1], reverse=True)
    base_mask = np.zeros_like(canny)
    #We generate a new image formed only by the contours that are at least area_thr of the size of the largest contour
    largest_area = sorted_contours[0][1]
    for contour in sorted_contours:
        if contour[1] >= largest_area * area_thr:
            cv2.fillConvexPoly(base_mask, contour[0], [256, 256, 256])
        else:
            break
    #Morphological operations to get rid of potential holes in the mask
    final_mask = cv2.erode(cv2.dilate(base_mask, None, iterations=5), None, iterations=8)
    return final_mask


def are_paintings_horizontal(rectangles):
    if len(rectangles) == 1:
        return True
    min_points = []
    max_points = []
    centers = []
    for rectangle in rectangles:
        center = rectangle[0]
        box = cv2.boxPoints(rectangle)
        box = np.int0(box)
        min_left = math.inf
        max_left = 0
        for point in box:
            if point[0] <= min_left:
                min_left = point[0]
            if point[0] >= max_left:
                max_left = point[0]
        min_points.append(min_left)
        max_points.append(max_left)
        centers.append(center)
    min_points, max_points, centers = zip(*sorted(zip(min_points, max_points, centers)))
    for i in range(1, len(rectangles)):
        if centers[i][0] <= max_points[i-1]:
            return False
    return True


def process_rectangles(rectangles):
    if len(rectangles) == 1:
        return rectangles
    horizontal = are_paintings_horizontal(rectangles)
    if horizontal:
        return sorted(rectangles, key=lambda x: x[0][0])
    else:
        return sorted(rectangles, key=lambda x: x[0][1])


def get_frames_from_mask(mask, area_thr = 0.03):
    img_area = mask.shape[0] * mask.shape[1]
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_points = contours[0]
    rects = []
    for contour in contour_points:
        rect = cv2.minAreaRect(contour)
        box_area = rect[1][0] * rect[1][1]
        #We discard potential false positives
        if box_area > img_area * area_thr:
            rects.append(rect)

    return process_rectangles(rects)


def get_paintings_from_frames(img, rects):
    subimages = []
    for rect in rects:
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        # corrdinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        subimage = cv2.warpPerspective(img, M, (width, height))

        angle = rect[2]

        if 0.0 >= angle > -45.0:
            angle = angle * -1.0
        elif angle < -45.0:
            angle = angle * -1.0 + 90.0

        if angle > 90:
            subimage = np.rot90(subimage)

        subimages.append(subimage)
    return subimages


def get_box(rectangle):
    return cv2.boxPoints(rectangle).tolist()


def correct_angle(angle):
    if 0.0 >= angle > -45.0:
        return angle * -1.0
    elif angle < -45.0:
        return angle * -1.0 + 90.0


def get_median_angle(rects):
    angles = [rect[2] for rect in rects]
    angles = np.array(angles)
    return correct_angle(np.median(angles))

