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
    result = sum(sum(np.diff(np.concatenate((l, r), axis=1), n=1)) for l, r in zip(ls, rs))
    return result


def hellinger(ls, rs):
    result = sum(sum(np.sqrt(l * r)) for l, r in zip(ls, rs))
    return result


def canberra(ls, rs):
    result = sum(distance.canberra(l, r) for l, r in zip(ls, rs))
    return result