from dataset import HistDataset
from distance import euclidean
from utils import calc_similarities, get_tops

QS1 = HistDataset("datasets/qsd1_w1")
DB = HistDataset("datasets/DDBB")

sims = calc_similarities(euclidean, DB, QS1, True)
tops = get_tops(sims, 4)

print(sims[0, tops[0]])
