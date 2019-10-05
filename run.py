from dataset import HistDataset
from distance import euclidean
from utils import calc_similarities, get_tops
import matplotlib.pyplot as plt
QS1 = HistDataset("datasets/qsd1_w1")
DB = HistDataset("datasets/DDBB")

sims = calc_similarities(euclidean, DB, QS1, True)
tops = get_tops(sims, 4)

# print(sims[0, tops[0]])
# print(DB[87][1])
R=DB[87][0]
G=DB[87][1]
B=DB[87][2]
plt.plot(R,'r',G,'g',B,'b')
plt.ylabel('Histogram')
plt.show()