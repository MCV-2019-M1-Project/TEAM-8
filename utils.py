import numpy as np
from tqdm.auto import tqdm
import pickle


def calc_similarities(measure, db, qs, show_progress=False):
    """
    Returns an array of size (qs_size x db_size)
    where arr[i,j] = similarity between
    i-th image in queryset and j-th image in database
    """
    def compute_one(hist):
        return [measure(hist, db_hist) for db_hist in db]
    generator = tqdm(qs) if show_progress else qs
    return np.array([compute_one(hist) for hist in generator])


def get_tops(similarities, k):
    """
    Returns an array of size (qs_size x k)
    where arr[i,j] is the index of j-th closest image in the database
    to i-th image in the queryset
    """
    tops = similarities.argsort(axis=1)[:, :k]
    return tops


def get_groundtruth(path):
    pklFile = open(path, "rb")
    groundTruth = pickle.load(pklFile)
    return [[item[0][1] for item in groundTruth]]

# TODO(Marc): Evaluation with MAP@k
