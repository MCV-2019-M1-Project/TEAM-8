import ml_metrics as metrics
from dataset import Dataset
from dataset import HistDataset
import distance as dist
from utils import calc_similarities, get_tops, get_groundtruth, normalize_hist
import matplotlib.pyplot as plt
'''For background removal vis HLS values go to dataset.py and check True, didn't have time to put it here cleanly'''

groundTruth = get_groundtruth("datasets/qsd1_w1/gt_corresps.pkl")

QS = [normalize_hist(qs_hist) for qs_hist in HistDataset("datasets/qsd1_w1")]
# QS2 = Dataset("datasets/qsd2_w1")
DB = [normalize_hist(db_hist) for db_hist in HistDataset("datasets/DDBB")]

k = 10

sims = calc_similarities(dist.canberra, DB, QS, True)
tops = get_tops(sims, k)
mapAtK = metrics.mapk(groundTruth, tops, k)

print(str(tops[0]))
print(str(tops[1]))
print(str(tops[2]))

print("Map@k is " + str(mapAtK))

#If you want to display any specific histogram
# R=DB[87][0]
# G=DB[87][1]
# B=DB[87][2]
# plt.plot(R,'r',G,'g',B,'b')
# plt.ylabel('Histogram')
# plt.show()
