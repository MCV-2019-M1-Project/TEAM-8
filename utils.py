import numpy as np
from tqdm.auto import tqdm
import pickle
import cv2


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


def background_removal(dataset):

    for i in range(dataset.__len__()):
        denoised_dataset=cv2.GaussianBlur(dataset[i][:, :, :], (5,5),0)
    return denoised_dataset

def edgedetector (color_channel):
    sobelX = cv2.Sobel(color_channel,cv2.CV_16S,1,0)
    sobelY = cv2.Sobel(color_channel, cv2.CV_16S,0,1)
    result= np.hypot(sobelX,sobelY)