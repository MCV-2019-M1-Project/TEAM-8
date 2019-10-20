import text_removal
import cv2
from utils import get_groundtruth, get_mean_IoU
import numpy as np


QS1 = [
    text_removal.getpoints2(im) for im in text_removal.text_remover("datasets/qsd1_w2")
]
boundingxys = [element.boundingxy for element in QS1]
drawings = [element.drawing for element in QS1]

gt = np.asarray(get_groundtruth("datasets/qsd1_w2/text_boxes.pkl")).squeeze()
mean_IoU = get_mean_IoU(gt, boundingxys)

print("Mean Intersection over Union: ", mean_IoU)

for im in range(len(drawings)):
    cv2.imwrite("outputs/" + str(im) + ".png", drawings[im])
