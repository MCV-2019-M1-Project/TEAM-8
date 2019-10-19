import numpy as np
from tqdm.auto import tqdm
import pickle
import cv2
from mask_metrics import MaskMetrics
import distance as dist


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
        result += dist.intersection_over_union(gts[x], preds[x])
    return result / len(preds)


def getgradient(img):
    x, y = np.gradient(img)
    return np.hypot(x, y).astype(np.uint8)


def lapl_at_index(source, index):
    i,j = index
    val = (4 * source[i,j])    \
           - (1 * source[i+1, j]) \
           - (1 * source[i-1, j]) \
           - (1 * source[i, j+1]) \
           - (1 * source[i, j-1])
    return val