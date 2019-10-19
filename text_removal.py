import dataset
import glob
import numpy as np
import cv2
class text_remover(dataset.Dataset):
    
    def __init__(self, path):
        self.paths = sorted(glob.glob(f"{path}/*.jpg"))
        self.data = [cv2.imread(path) for path in self.paths]

    def __getitem__(self, idx):
        return super().__getitem__(idx)  

def getpoints(im):

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
    return final


def getpoints2(im):
    # print(im.shape[0])
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
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
    kernel1= np.ones((6,6),np.uint8)
    kernel = np.ones((int((im.shape[0])/350),int((im.shape[1])/35)), np.uint8)

    denoised = cv2.erode(mask1, kernel, iterations=5)
    denoised = cv2.erode(denoised, kernel, iterations=1)
    denoised = cv2.dilate(denoised, kernel, iterations=3)
    denoised = cv2.dilate(denoised, kernel, iterations=2)

    canny_output = cv2.Canny(denoised, 200, 255)
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing[:,:,0] = gray
    max_area=0
    max=0
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        area = boundRect[i][2]*boundRect[i][3]
        if (area > max_area) & (boundRect[i][3] < boundRect[i][2]):
            max=boundRect[i]
            max_area=area
        boundRect[i] = max

    for i in range(len(contours)):
        # cv2.drawContours(drawing, contours_poly, i, (0, 255, 0))
        cv2.rectangle(drawing, (int(boundRect[-1][0]), int(boundRect[-1][1])), \
          (int(boundRect[-1][0]+boundRect[-1][2]), int(boundRect[-1][1]+boundRect[-1][3])), (0, 255, 0), 2)

    class Result:
        def __init__(self, boundingxy, drawing):
            self.boundingxy = boundingxy
            self.drawing = drawing

    boundingxy=[boundRect[-1][0],boundRect[-1][1],boundRect[-1][0] + boundRect[-1][2], boundRect[-1][1] + boundRect[-1][3]]
    #Coordinates [tlx,tly,brx,bry]

    return Result(boundingxy, drawing)


def getpoints3(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imagehigh = gray > 200
    imagelow = gray < 50
    imager = im[:,:,0]
    imageg = im[:,:,1]
    imageb = im[:,:,2]
    output = np.equal(imager,imageg,dtype=int)
    output2=np.equal(imager,imageb,dtype=int)
    output1=np.equal(imageb,imageg,dtype=int)
    output3=output&output2
    output4=output3&output1
    equalhigh=output4 & imagehigh
    equallow=output4 & imagelow

    kernel = np.ones((int((im.shape[0]) / 350), int((im.shape[1]) / 20)), np.uint8)
    equallow = 255 * equallow
    equallow = equallow.astype(np.uint8)
    equalhigh = 255 * equalhigh
    equalhigh = equalhigh.astype(np.uint8)

    equalhigh = cv2.erode(equalhigh, kernel, iterations=1)
    equalhigh = cv2.dilate(equalhigh, kernel, iterations=6)
    equalhigh = cv2.erode(equalhigh, kernel, iterations=7)
    if equalhigh.any() > 0:
        print('imatge blanca')
        #TODO: Probar amb el threshold aqui, ja que saps si es blanca o negre amb un 1/30 de error.
        final = 255 * imagehigh
        final = final.astype(np.uint8)
    else :
        print('imatge negre')
        final=None

    return final

