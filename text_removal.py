import dataset
import glob
import numpy as np
import cv2
import os
class text_remover(dataset.Dataset):
    
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))

    def __getitem__(self, idx):
       
        return super().__getitem__(idx)  

def getpoints(im):

    #im = cv2.GaussianBlur(img, (5, 5), 0)
   #Crisis solved
    #hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    #hsv = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #ret, thresh = cv2.threshold(
    #    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #ret, thresh = cv2.threshold(
    #gray, 220,255, cv2.THRESH_TOZERO_INV)
    ret, thresh1 = cv2.threshold(
    im[:,:,0], 200,255, cv2.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    sure_bg1 = cv2.erode(thresh1, kernel, iterations=3)

    ret, thresh2 = cv2.threshold(
    im[:,:,1], 200,255, cv2.THRESH_BINARY)
 
    sure_bg2 = cv2.erode(thresh2, kernel, iterations=3)

    ret, thresh3 = cv2.threshold(
    im[:,:,2], 200,255, cv2.THRESH_BINARY)
    sure_bg3 = cv2.erode(thresh3, kernel, iterations=3)
    
    final=sure_bg1-(sure_bg3-sure_bg1-sure_bg2)
    

    #cv2.imshow('try',sure_bg[1])
    #cv2.waitKey(0)
        
    #cv2.waitkey(0)
    return final
    #return sure_bg1,sure_bg2,sure_bg3
    #return hsv

def getpoints2(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(
    gray, 200, 255, cv2.THRESH_BINARY)
    edges=cv2.Canny(im,200,220)
    kerneledges = np.array([[0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1],
                            [0, 1,1,1,1,1,1,1,1,1]], np.uint8)
    kernel = np.ones((20), np.uint8)
    #edge = cv2.erode(thresh, kerneledges, iterations=3)
    denoised = cv2.erode(thresh, kernel, iterations=3)
    print(type(im))

    return denoised
  

    


