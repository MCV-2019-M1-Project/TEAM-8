from dataset import Dataset

import cv2
import numpy as np

"""
This is to remove the background it basically searches for the key points of the rectangle and removes the background.

It just works fine with this particular scenario.

P2  P1

P3  P0


"""

QS2 = Dataset("datasets/qsd2_w1")
img = cv2.GaussianBlur(QS2[1], (5, 5), 0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

i, j = np.where(thresh == 255)

# for i in range(2000):
#     print(thresh[100,i])
k, d = np.where(thresh[:, 0:100] == 255)  # To actually find the points 2 and 3


points = np.zeros((4, 2))
points[0] = (j[-1], i[-1])
points[1] = (j[0], i[0])
points[2] = (d[0], k[0])
points[3] = (d[-1], k[-1])

print(points)

image_countours = cv2.fillPoly(thresh, np.int32([points]), (255, 255, 255), 8, 0, None)
image_countours = cv2.resize(image_countours, (1920, 1080))
cv2.imshow("dd", image_countours)
cv2.waitKey(0)

# print("dd")
# print(k)
# plt.imshow(thresh)
# plt.show()
# cv2.waitKey(0)
