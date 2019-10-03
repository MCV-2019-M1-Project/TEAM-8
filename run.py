import ml_metrics as metrics
from dataset import HistDataset
from distance import euclidean
from utils import calc_similarities, get_tops, get_groundtruth

QS1 = HistDataset("datasets/qsd1_w1")
DB = HistDataset("datasets/DDBB")
groundTruth = get_groundtruth("datasets/qsd1_w1/gt_corresps.pkl")

k = 4

sims = calc_similarities(euclidean, DB, QS1, True)
tops = get_tops(sims, k)
mapAtK = metrics.mapk(groundTruth, tops, k)

print(sims[0, tops[0]])
print("Map@k is " + str(mapAtK))
