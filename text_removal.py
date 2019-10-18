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
    # print(im.shape[0])
    imager=im[:,:,0]
    imageg=im[:,:,1]
    imageb=im[:,:,2]
    output=np.equal(imager,imageg,dtype=int)
    output2=np.equal(imager,imageb,dtype=int)
    output1=np.equal(imageb,imageg,dtype=int)
    output3=output&output2
    output4=output3&output1
    mask=255*output4
    mask1=mask.astype(np.uint8)
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(
    # gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((int((im.shape[0])/350),int((im.shape[1])/50)), np.uint8)
    #edge = cv2.erode(thresh, kerneledges, iterations=3)
    denoised = cv2.erode(mask1, kernel, iterations=3)
    denoised = cv2.erode(denoised, kernel, iterations=3)
    denoised = cv2.dilate(denoised, kernel, iterations=3)
    denoised = cv2.dilate(denoised, kernel, iterations=3)
    return denoised


def getpoints3(im):
    histr=cv2.calcHist([im], [0], None, [256], [0, 256]),
    histg=cv2.calcHist([im], [1], None, [256], [0, 256]),
    histb=cv2.calcHist([im], [2], None, [256], [0, 256]),
    histrg=[]*256
    histrb=[]*256
    histresultat=[]*256
    print(histr[0][0])
    print(histb[0][0])
    print(histg[0][0])
    histrg=np.equal(histr,histg)
    histrb=np.equal(histr,histb)
    histresultat=histrb&histrg
    # for i in range(len(histg)):
    #     if int(histr[0][i]) == int(histg[0][i]):
    #         histrg[0][i] = histr[0][i]
    #     else:
    #         histrg[0][i] = 0
    #     if int(histr[0][i]) == int(histb[0][i]):
    #         histrb[0][i] = histr[0][i]
    #     else:
    #         histrb[0][i] = 0
    #
    #     if int(histrg[0][i]) == int(histrb[0][i]):
    #         histresultat[0][i] = int(histrb[0][i])
    #     else:
    #         histresultat[0][i]==0
        # histrg[i]=histr[i]&histg[i]
        # histrb[i]=histr[i]&histb[i]
        # histresultat[i]=histrb[i]&histrg[i]
    # for i in range(len(histg)):
    #     if histr[0].index(i) == histg[0].index(i):
    #         histrg[0][i] = histr[0][i]
    #     else:
    #         histrg[0][i] = 0
    #     if int(histr[0][i]) == int(histb[0][i]):
    #         histrb[0][i] = histr[0][i]
    #     else:
    #         histrb[0][i] = 0
    #
    #     if int(histrg[0][i]) == int(histrb[0][i]):
    #         histresultat[0][i] = int(histrb[0][i])
    #     else:
    #         histresultat[0][i] == 0

    return histg
def getpoints4(im):
    imager=im[:,:,0]
    imageg=im[:,:,1]
    imageb=im[:,:,2]
    output=np.equal(imager,imageg,dtype=int)
    output2=np.equal(imager,imageb,dtype=int)
    output1=np.equal(imageb,imageg,dtype=int)
    output3=output&output2
    output4=output3&output1

    return output4






def getimg(im):
    return im