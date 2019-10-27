import cv2
from skimage import feature

def get_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return feature.local_binary_pattern(gray, 8, 1, method="default")