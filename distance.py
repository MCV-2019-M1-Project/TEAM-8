from scipy.spatial import distance
import numpy as np

# Each measure should take two lists of histograms
# and return a final score (as a single number)


def euclidean(ls, rs):
    result = sum(distance.euclidean(l, r) for l, r in zip(ls, rs))
    return result


def l_one(ls, rs):
    result = sum(distance.cityblock(l, r) for l, r in zip(ls, rs))
    return result


def x2(ls, rs):
    result = 0
    for l, r in zip(ls, rs):
        for lind, rind in zip(l, r):
            if lind + rind != 0:
                result += pow(lind - rind, 2) / (lind + rind)

    return result


def h_intersection(ls, rs):
    result = sum(
        sum(np.diff(np.concatenate((l, r), axis=1), n=1)) for l, r in zip(ls, rs)
    )
    return result


def hellinger(ls, rs):
    result = sum(sum(np.sqrt(l * r)) for l, r in zip(ls, rs))
    return result


def canberra(ls, rs):
    result = sum(distance.canberra(l, r) for l, r in zip(ls, rs))
    return result


def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA = np.asarray(boxA).squeeze()
    boxB = np.asarray(boxB).squeeze()
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def compare_hogs(hog1, hog2):
    return hog2 - hog1
