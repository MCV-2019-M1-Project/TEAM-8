"""
Example usage: python run.py task2
"""
import pickle
import numpy as np
import cv2
import fire
import ml_metrics as metrics
from tqdm.auto import tqdm

import text_removal
from utils import get_groundtruth, get_mean_IoU, dump_pickle, get_pickle, get_gt_text, compute_lev, get_hog_histograms, denoise_image
from dataset import Dataset, MaskDataset, HistDataset, MultiHistDataset, BBox
import distance as dist


from utils import (
    calc_similarities,
    get_tops,
    get_mask_metrics,
    calc_multi_similarities,
    get_multi_tops,
)


def find_img_corresp(QS, GT, DB, k):
    similarieties = calc_similarities(dist.canberra, DB, QS, True)
    tops = get_tops(similarieties, k)
    dump_pickle("result.pkl", tops)
    mapAtK = metrics.mapk(GT, tops, k)
    print("Map@k is " + str(mapAtK))


def find_multi_img_corresp(QS, GT, DB, k):
    similarieties = calc_multi_similarities(dist.canberra, DB, QS, True)
    tops = get_multi_tops(similarieties, k, len(DB))
    mapAtK = metrics.mapk(GT, tops, k)
    print("Map@k is " + str(mapAtK))


def find_multi_img_corresp_keep(QS, DB, k):
    def calc_multi_similarities_keep(measure, db, qs, show_progress=False):
        def compute_one(hists):
            result = [
                np.array([measure(hist, db_hist) for db_hist in db]) for hist in hists
            ]
            return result

        generator = tqdm(qs) if show_progress else qs
        return [compute_one(hist) for hist in generator]

    def get_multi_tops_keep(sims, k):
        sims = list(
            map(lambda sims_pic: [sims.argsort()[:k].tolist() for sims in sims_pic], sims)
        )
        return sims

    sims = calc_multi_similarities_keep(dist.canberra, DB, QS, True)
    tops = get_multi_tops_keep(sims, k)
    return tops


def eval_masks(QS, MS_GT):
    if len(QS) != len(MS_GT):
        raise ValueError("Query set size doesn't match ground truth size")
    mask_dataset = Dataset("datasets/qsd2_w1", masking=True)
    print("Predicting masks")
    predicted_masks = [mask_dataset.get_mask(i) // 255 for i in tqdm(range(len(QS)))]
    print("Calculating mask metrics")
    mask_metrics = get_mask_metrics(predicted_masks, MS_GT, True)

    print("Precision: " + str(mask_metrics["precision"]))
    print("Recall: " + str(mask_metrics["recall"]))
    print("F1-score: " + str(mask_metrics["f1_score"]))


class Solution:
    def __init__(
        self,
        DDBB="datasets/DDBB",
        QSD1_W1="datasets/qsd1_w1",
        QSD2_W1="datasets/qsd2_w1",
        QSD1_W2="datasets/qsd1_w2",
        QSD2_W2="datasets/qst2_w2",
        QSD1_W3="datasets/qsd1_w3",
        QSD2_W3="datasets/qsd2_w3",
        QST1_W3="datasets/qst1_w3",
        QST2_W3="datasets/qst2_w3",
    ):
        self.QSD1_W1 = QSD1_W1
        self.QSD2_W1 = QSD2_W1
        self.QSD1_W2 = QSD1_W2
        self.QSD2_W2 = QSD2_W2
        self.QSD1_W3 = QSD1_W3
        self.QSD2_W3 = QSD2_W3
        self.QST1_W3 = QST1_W3
        self.QST2_W3 = QST2_W3
        self.DDBB = DDBB

    def task2(self, k=10):
        QS2 = HistDataset(self.QSD1_W3, method="combo", masking=False, bbox=True, multires=4, denoise=True, texture="LBP")
        GT = get_groundtruth("datasets/qsd1_w3/gt_corresps.pkl")
        print(f"Computing normalized histograms for {self.DDBB}")
        DB = list(tqdm(HistDataset(self.DDBB, masking=False, method="combo", multires=4, texture="LBP")))
        print("Analyzing QS2")
        find_img_corresp(QS2, GT, DB, k)

    def task_w2(self):
        print("Loading images...")
        QS1 = text_removal.text_remover(self.QSD1_W3)
        DDBB = text_removal.text_remover(self.DDBB)

        print("\nComputing histograms...")
        hogs_qs = get_hog_histograms(QS1)
        hogs_ddbb = get_hog_histograms(DDBB)

        print("\nComputing pairs...")
        predictions = []

        for x in tqdm(range(len(hogs_qs))):
            min_diff_pos = 0
            min_diff_value = 999999999999999

            for y in range(len(hogs_ddbb)):
                diff = sum(np.abs(hogs_qs[x] - hogs_ddbb[y]))
                if diff < min_diff_value:
                    min_diff_value = diff
                    min_diff_pos = y
            predictions.append(min_diff_pos)

        print("\n", predictions)
        print("Computing bounding boxes and reading text...")
        results = [text_removal.getpoints2(im) for im in tqdm(QS1)]

        bbs_predicted = [element.boundingxy for element in results]
        texts_predicted = [element.text for element in results]
        images = [element.drawing for element in results]

        bbs_gt = np.asarray(get_groundtruth(f"{self.QSD1_W3}/text_boxes.pkl")).squeeze()
        mean_iou = get_mean_IoU(bbs_gt, bbs_predicted)
        print("\nMean Intersection over Union: ", mean_iou)

        texts_gt = get_gt_text(f"{self.QSD1_W3}")
        mean_lev = compute_lev(texts_gt, texts_predicted)
        print("Mean Levenshtein distance: ", mean_lev)

        for im in range(len(images)):
            cv2.imwrite("outputs/" + str(im) + ".png", images[im])

    def task4(self):
        print("Computing bounding boxes")
        QS1 = [
            text_removal.getpoints2(im)
            for im in tqdm(text_removal.text_remover(self.QSD1_W2))
        ]
        boundingxys = [element.boundingxy for element in QS1]
        drawings = [element.drawing for element in QS1]

        gt = np.asarray(get_groundtruth(f"{self.QSD1_W2}/text_boxes.pkl")).squeeze()
        mean_IoU = get_mean_IoU(gt, boundingxys)

        print("Mean Intersection over Union: ", mean_IoU)

        for im in range(len(drawings)):
            cv2.imwrite("outputs/" + str(im) + ".png", drawings[im])

    def task5(self):
        # Get text box pkl
        QS = [
            text_removal.getpoints2(im)
            for im in text_removal.text_remover(self.QSD1_W2)
        ]
        boundingxys = [[element.boundingxy] for element in QS]
        with open("QSD1/text_boxes.pkl", "wb") as f:
            pickle.dump(boundingxys, f)

        # Get text box pngs TODO bbox
        QS1 = HistDataset("datasets/qsd1_w2", bbox=True, multires=2)
        predicted_masks = [QS1.get_mask(idx) for idx in range(len(QS1))]
        for i, img in enumerate(predicted_masks):
            filename = "QSD1/boxes/" + f"{i:05d}" + ".png"
            cv2.imwrite(filename, img)

        gt = np.asarray(get_groundtruth("datasets/qsd1_w2/text_boxes.pkl")).squeeze()
        mean_IoU = get_mean_IoU(gt, boundingxys)
        print(f"Mean IoU: {mean_IoU}")

    def task6(self, k=10):
        QS = [  # noqa
            hists
            for hists in tqdm(MultiHistDataset(self.QSD2_W3, masking=True, bbox=True, multires=4, method="color", texture="LBP", denoise=True))
        ]
        GT = get_pickle("datasets/qsd2_w3/gt_corresps.pkl")
        DB = list(tqdm(HistDataset(self.DDBB, masking=False, multires=4, method="color", texture="LBP")))  # noqa
        tops = find_multi_img_corresp_keep(QS, DB, k)
        dump_pickle("result_qst2.pkl", tops)
        mapAtK = metrics.mapk(GT, tops, k)
        print("Map@k is " + str(mapAtK))
        exit()
        with open("outputs/resutls.pkl", "wb") as f:
            pickle.dump(tops, f)
        print(tops)

        # Generate pngs
        QS1 = Dataset(self.QSD2_W2, masking=True, bbox=True)
        for i in range(len(QS1)):
            im = QS1.get_mask(i)
            cv2.imwrite("outputs/" + str(i) + ".png", im)
        text_boxes = [BBox().get_bbox_cords(QS1[i]) for i in range(len(QS1))]
        with open("outputs/text_boxes.pkl", "wb") as f:
            pickle.dump(text_boxes, f)

    def eval_masks(self):
        QS = Dataset(self.QSD2_W1, masking=True)
        GT = MaskDataset(self.QSD2_W1)
        eval_masks(QS, GT)


if __name__ == "__main__":
    fire.Fire(Solution)
