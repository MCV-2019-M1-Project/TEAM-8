import ml_metrics as metrics
import cv2
from dataset import HistDataset
from dataset import MaskDataset
import distance as dist
import text_removal
#from utils import (
#    calc_similarities,
#    get_tops,
#    get_groundtruth,
#    normalize_hist,
#    get_mask_metrics,
#)
namedataset=[text_removal.getpoints(im) for im in text_removal.text_remover("datasets/qsd1_w2")]

    
#Fer el threshold per colors, ja que el valor de igual nomes el tenen el retols.
image_countours=cv2.resize(namedataset[9],(1280,
                                           700))
cv2.imshow('try', image_countours)
cv2.waitKey(0)


#def find_img_corresp(QS, groundTruth, masking):
#    sims = calc_similarities(dist.canberra, DB, QS, True)
#    tops = get_tops(sims, k)
#    mapAtK = metrics.mapk(groundTruth, tops, k)

#    print(str(tops[0]))
#    print(str(tops[1]))
#    print(str(tops[2]))
#    print("Map@k is " + str(mapAtK))

#    if masking:
#        gt_masks = MaskDataset("datasets/qsd2_w1")
#        mask_dataset = HistDataset("datasets/qsd2_w1", masking=True)
#        predicted_masks = [cv2.threshold(mask_dataset.get_mask(i), 128, 1, cv2.THRESH_BINARY)[1] for i, item in enumerate(mask_dataset)]

#        mask_metrics = get_mask_metrics(predicted_masks, gt_masks, True)

#        print("Precision: " + str(mask_metrics["precision"]))
#        print("Recall: " + str(mask_metrics["recall"]))
#        print("F1-score: " + str(mask_metrics["f1_score"]))


#groundTruth1 = get_groundtruth("datasets/qsd1_w1/gt_corresps.pkl")
#groundTruth2 = get_groundtruth("datasets/qsd2_w1/gt_corresps.pkl")

#QS1 = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
#QS2 = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd2_w1", masking=True)]
#DB = [normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB", masking=False)]

#k = 10

#print("Analyzing QS1")
#find_img_corresp(QS1, groundTruth1, False)

#print("Analyzing QS2")
#find_img_corresp(QS2, groundTruth2, True)






