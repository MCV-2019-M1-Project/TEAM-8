import dataset
import glob
import numpy as np
import cv2
class text_remover(dataset.Dataset):
    
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))

    def __getitem__(self, idx):
       
        return super().__getitem__(idx)  

def getpoints(im):

    #im = cv2.GaussianBlur(img, (5, 5), 0)
        #Identity Crisis
    #Whos Eddie
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(
    #    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(
    #gray, 220,255, cv2.THRESH_TOZERO_INV)
    ret, thresh = cv2.threshold(
    gray, 200,255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    sure_bg = cv2.erode(thresh, kernel, iterations=3)
    #cv2.imshow('try',sure_bg[1])
    #cv2.waitKey(0)
        
    #cv2.waitkey(0)
    return sure_bg


    


