import pickle

import cv2
import numpy as np
from tqdm.auto import tqdm

from mask_metrics import MaskMetrics
import distance as dist
import Levenshtein as lev
import glob
from skimage import feature


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
    tops = similarities.argsort(axis=1)[:, :k]
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
    for x in range(len(preds)):
        res = lev.distance(gts[x], preds[x])
        if res != 0:
            print("  Lev distance at", x, ":", res)
        result += res
    return result / len(preds)


def get_gt_text(path):
    paths = sorted(glob.glob(f"{path}/*.txt"))
    result = []
    for path in paths:
        result.append(open(path, "r").read().split("'")[1])
    return result


def get_hog_histogram(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    descriptors = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1", feature_vector=False)
    return descriptors.ravel()


def get_hog_histograms(imgs):
    hogs_imgs = [get_hog_histogram(imgs[x]) for x in tqdm(range(len(imgs)))]
    return hogs_imgs

def denoise_image(img, method="Gaussian"):
    if method == "Gaussian":
        return cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)
    elif method == "Median":
        #Median gives best results for qsd1 at least
        return cv2.medianBlur(img, 3)
    elif method == "bilateral":
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif method == "FastNl":
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

