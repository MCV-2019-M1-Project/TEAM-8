from dataset import HistDataset
from distance import euclidean
from main import calc_similarities, get_tops

QS1 = HistDataset("datasets/queryset1")
DB = HistDataset("datasets/database")

sims = calc_similarities(euclidean, DB, QS1, True)
tops = get_tops(sims, 4)

sims[0, tops[0]]