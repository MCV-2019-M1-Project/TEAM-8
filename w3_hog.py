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

def get_imgs(files_path, extension ="jpg"):
    paths = sorted(glob.glob(f"{files_path}/*." + extension))
    return [cv.imread(path) for path in tqdm(paths)]


# TODO: Check if best option
def denoise_imgs(img):
    return cv.medianBlur(img, 3)

def main():
    #K parameter for map@k
    k = 3
    # Get images and denoise query set.
    print("Getting and denoising images...")
    qs = get_imgs("datasets/qsd1_w4")
    db = get_imgs("datasets/DDBB")
    qs_denoised = [denoise_imgs(img) for img in tqdm(qs)]

    #Separating paitings inside images to separate images
    qs_split = [background_remover.remove_background(img) for img in qs_denoised]

    print("\nComputing histograms...")
    hogs_qs = [[utils.get_hog_histogram(painting) for painting in img] for img in qs_split]
    hogs_ddbb = utils.get_hog_histograms(db)

    print("\nComputing distances")
    distances = []

    #Generating distances between qs images and db images
    for im in tqdm(hogs_qs):
        current_im = []
        for painting_hog in im:
            current_pt = []
            for db_hog in hogs_ddbb:
                current_pt.append(sum(np.abs(painting_hog - db_hog)))
            current_im.append(current_pt)
        distances.append(current_im)

    print("Done calculating hogs")

    #Generating predictions
    predictions = []

    for im in distances:
        current_im = []
        for painting_dst in im:
            current_im.append(utils.list_argsort(painting_dst)[:k])
        predictions.append(current_im)

    hypo = []
    for im in predictions:
        current_im = []
        for painting in im:
            for pred in painting:
                current_im.append(pred)
        hypo.append(current_im)

    #Generate map@k
    gt = utils.get_pickle("datasets/qsd1_w4/gt_corresps.pkl")
    mapAtK = metrics.mapk(gt, hypo, k)
    print("\nMap@ " + str(k) + " is " + str(mapAtK))

main()
