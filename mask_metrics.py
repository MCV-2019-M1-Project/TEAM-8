import numpy as np


class MaskMetrics:
    def __init__(self, pred, gt):
        if pred.shape[0] != gt.shape[0] or pred.shape[1] != gt.shape[1]:
            raise Exception("The two arrays should be the same size")
        self.results = np.zeros([2, 2])
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                self.results[pred[i, j], gt[i, j]] += 1

    def precision(self):
        true_positives = self.results[1, 1]
        false_positives = self.results[1, 0]
        return true_positives / float(true_positives + false_positives)

    def recall(self):
        true_positives = self.results[1, 1]
        false_negatives = self.results[0, 1]
        return true_positives / float(true_positives + false_negatives)

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * ((precision * recall) / float(precision + recall))
