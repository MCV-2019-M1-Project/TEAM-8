import glob

import numpy as np
import cv2


class Dataset:
    def __init__(self, path, mask):
        self.paths = glob.glob(f"{path}/*.jpg")
        print(len(self.paths))

    def __getitem__(self, idx):
        # return cv2.imread(self.paths[idx],cv2.IMREAD_GRAYSCALE)
        return cv2.imread(self.paths[idx])
    def __len__(self):
        return len(self.paths)


class HistDataset(Dataset):
    """
        Calculates the histogram of the images
        and applies a mask on from RGB to HLS on the Histogram calculation
    """


    def __init__(self, *args, caching=True, **kwargs):
        self.caching = caching
        if caching:
            self.cache = dict()
        super().__init__(*args, **kwargs)
        print(args)
        self.masking = args[1]

    def calc_hist(self, img):
        def calc_mask():

            if False:
                result = np.zeros(img.shape[:2], dtype="uint8")

                hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
                hls_split = cv2.split(hls)

                lu = np.asarray(hls_split[1])
                sa = np.asarray(hls_split[2])

                sat_thresh = 100
                lum_tresh = 75

                result[(sa > sat_thresh) & (lu < lum_tresh)] = 1

                #return result

                pts = cv2.findNonZero(result).squeeze().tolist()

                num = 50

                pts_up_left = np.asarray(sorted(pts, key=lambda x: (x[0] + x[1]))[1:num])

                pts_bottom_left = np.asarray(sorted(pts, key=lambda x: (-(x[1] / (x[0] + 1))))[1:num])

                pts_up_right = np.asarray(sorted(pts, key=lambda x: (-(x[0] / (x[1] + 1))))[1:num])

                pts_bottom_right = np.asarray(sorted(pts, key=lambda x: (-x[0] + -x[1]))[1:num])

                a = 9999
                pt_up_left = pts_up_left[0]
                pt_up_right = pts_up_right[0]

                for y in range(1000):
                    for x in range(1000):
                        a = abs(pts_up_left[y][1] - pts_up_right[x][1])
                        if a < 75:
                            pt_up_left = pts_up_left[y]
                            pt_up_right = pts_up_right[x]
                            break
                    if a < 75:
                        break

                diff = np.abs(pts_bottom_right[:][0] - pt_up_right[0])
                pt_bottom_right = pts_bottom_right[np.where(diff == np.amin(diff))[0]][0]

                diff = np.abs(pts_bottom_left[:][1] - pt_bottom_right[1])
                pt_bottom_left = pts_bottom_left[np.where(diff == np.amin(diff))[0]][0]

                points = np.array((pt_up_right, pt_up_left, pt_bottom_left, pt_bottom_right))
                res2 = cv2.fillPoly(result, np.int32([points]), 1, 8, 0, None)

                """im = cv2.circle(img, (pt_up_left[0], pt_up_left[1]), 10, (255, 0, 0), 3) #blue
                im = cv2.circle(img, (pt_up_right[0], pt_up_right[1]), 10, (0, 255, 0), 3) #green
                im = cv2.circle(img, (pt_bottom_right[0], pt_bottom_right[1]), 10, (255, 255, 255), 3) #white
                im = cv2.circle(img, (pt_bottom_left[0], pt_bottom_left[1]), 10, (0, 0, 255), 3) #red

                scale_percent = 25  # percent of original size
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)

                resized1 = cv2.resize(res2, dim, interpolation=cv2.INTER_AREA)
                resized2 = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

                cv2.imshow("a", resized1)
                cv2.imshow("img",resized2)
                cv2.waitKey()
                cv2.destroyAllWindows()"""

                return res2

            else :
                im = cv2.GaussianBlur(img, (5, 5), 0)

                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                i, j = np.where(thresh == 255)
                y1 = i[0]  # fist point
                x1 = j[0]
                y4 = i[-1]  # last point
                x4 = j[-1]

                k, d = np.where(thresh[:, 0:100] == 255)
                y2 = k[0]  # second point
                x2 = d[0]
                y3 = k[-1]  # third point
                x3 = d[-1]

                points = np.array([[11, 13],
                                   [14, 16],
                                   [17, 11],
                                   [12, 15]]).astype('int32')

                points[1] = (j[0], i[0])
                points[2] = (d[0], k[0])
                points[3] = (d[-1], k[-1])
                points[0] = (j[-1], i[-1])

                image_countours = cv2.fillPoly(thresh, np.int32([points]), (255, 255, 255), 8, 0, None)

                return image_countours

        """
        if background should be removed or not
        """
        if self.masking:
            mask = calc_mask()
        else:
            mask = None

        return np.array(
            [
                cv2.calcHist([img], [0], mask, [256], [0, 256]),
                cv2.calcHist([img], [1], mask, [256], [0, 256]),
                cv2.calcHist([img], [2], mask, [256], [0, 256]),
            ]
        )

    def _calculate(self, idx):
        self.cache[idx] = self.calc_hist(super().__getitem__(idx))
        return self.cache[idx]

    def __getitem__(self, idx):
        if self.caching:
            return self.cache[idx] if idx in self.cache else self._calculate(idx)
        return self.calc_hist(super().__getitem__(idx))
