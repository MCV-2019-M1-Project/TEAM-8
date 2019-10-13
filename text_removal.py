import dataset
import glob
import numpy as np
import cv2
class text_remover(dataset.Dataset):
    
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))
        
    def getpoints(self, img):

        im = cv2.GaussianBlur(img, (5, 5), 0)

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)
        print('here')
        #cv2.imshow('try',sure_bg[1])
        #cv2.waitKey(0)
        
        #cv2.waitkey(0)
        return sure_bg







    def __getitem__(self, idx):
        print('1')
        return self.getpoints(super().__getitem__(idx))


