import cv2
import matplotlib.pyplot as plt
import skimage.segmentation
import itertools
import numpy as np


def low_pass_filter(img):
    kernel = np.ones((5, 5), np.float32) / 100
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def morphological_processing(gray_im, thr, blur_method, post_thresholding):
    # Disk Structural elements
    S5 = skimage.morphology.disk(5)
    S2 = skimage.morphology.disk(2)
    S3 = skimage.morphology.disk(3)

    # Applying closing to the image
    closing = cv2.morphologyEx(gray_im, cv2.MORPH_CLOSE, S5)

    # Applying opening to the image
    opening = cv2.morphologyEx(gray_im, cv2.MORPH_OPEN, S5)

    # difference between closing of the image and opening
    Im_op_clo_diff = closing - opening
    if blur_method == 'gaussian':
        # Gaussian filter to smooth the image and reduce noise for a better binarization
        blur_img = cv2.GaussianBlur(Im_op_clo_diff, (5, 5), 3)

    elif blur_method == 'kernel':
        # 2D convolution filter to smooth the image and reduce noise for a better binarization
        blur_img = low_pass_filter(Im_op_clo_diff)

    # Binarization of the blurred image
    ret, th3 = cv2.threshold(blur_img, thr, 255, cv2.THRESH_BINARY)

    if post_thresholding == 'dilation':
        # dilation of the binarized image to reduce black areas
        processed_im = cv2.dilate(th3, S3, 2)
        dilated_im = processed_im
    elif post_thresholding == 'dilationErosion':
        # dilation and erosion of the binarized image
        dilated_im = cv2.dilate(th3, S3, iterations=3)
        processed_im = cv2.erode(dilated_im, S2, iterations=2)

    # find contours of the connected components in the binarized image
    # im2, cont, hierarchy = cv2.findContours(processed_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) PETA!!!!!!!!!
    # https://stackoverflow.com/questions/54734538/opencv-assertion-failed-215assertion-failed-npoints-0-depth-cv-32
    cont, _ = cv2.findContours(processed_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rgb = cv2.cvtColor(gray_im, cv2.COLOR_GRAY2RGB)

    return cont, rgb, dilated_im


def remove_overlapped(over_Rects, extt_rects, ind1, ind2):
    """
    Remove the rectangles that are overlapped and append a new rectangle that contains both
    """
    x1, y1, w1, h1 = extt_rects[ind1]
    x2, y2, w2, h2 = over_Rects
    x3 = min(x1, x2)
    y3 = min(y1, y2)
    w3 = max(x2 + w2 - x1, x1 + w1 - x2)
    h3 = max(h1, h2)
    extt_rects[ind1] = "False"
    extt_rects[ind2] = "False"
    new_rect = (x3, y3, w3, h3)
    extt_rects.append(new_rect)

    return extt_rects


def get_max_rectangle(rects):
    """
    Get the rectangle with biggest area
    """
    w_list = [item[2] for item in rects]
    h_list = [item[3] for item in rects]
    area_list = [a * b for a, b in zip(w_list, h_list)]
    if area_list:
        max_area = max(area_list)
        max_rect_ind = area_list.index(max_area)
        x, y, w, h = rects[max_rect_ind]
    else:
        return 0, 0, 0, 0
    return rects[max_rect_ind]


def erase_overlapped_regions(rects, im_w, im_h):
    """
    Remove the overlapped rectangles and merge the ones that are very near
    """

    index_list = []
    ext_rects = rects[:]
    overlapped = []
    #    a =
    for i in range(0, len(ext_rects)):
        overlapped[:] = []
        b = len(ext_rects)
        for jj in range(0, b):
            if ext_rects[i] != "False" and ext_rects[jj] != "False":

                x1, y1, w1, h1 = ext_rects[i]
                x2, y2, w2, h2 = ext_rects[jj]

                if x1 > x2 and abs(x1 - x2 - w2) < 0.040 * im_w and -0.01 * im_w < y1 - y2 < 0.01 * im_h:
                    overlapped.append(jj)

                else:
                    overlapped.append(-1)
            else:
                overlapped.append(-1)
                pass

        index_list[:] = []

        overlapped_list = [item != -1 for item in overlapped]
        elem_overlapped = list(itertools.compress(ext_rects, overlapped_list))
        for el in overlapped:
            if el != -1:
                index_list.append(el)
            else:
                pass
        for it in range(len(elem_overlapped)):
            if it != "False" and ext_rects[i] != "False":
                ext_rects = remove_overlapped(elem_overlapped[it], ext_rects, i, index_list[it])
    rectangles_no_overlapped = [item != "False" for item in ext_rects]
    ext_rects = list(itertools.compress(ext_rects, rectangles_no_overlapped))
    if False in rectangles_no_overlapped:
        ext_rects = erase_overlapped_regions(ext_rects, im_w, im_h)
    else:
        pass

    return ext_rects


def erase_inside_regions(rects):
    """
    Remove the rectangles contained in others
    """
    num_rects = len(rects) - 1
    ext_rects = rects[:]
    contains = []
    for i in range(num_rects + 1):
        contains[:] = []
        for j in range(num_rects + 1):
            x1, y1, w1, h1 = rects[i]
            x2, y2, w2, h2 = rects[j]

            if (x2 > x1 and x2 + w2 < x1 + w1 and y2 > y1 and y2 + h2 < y1 + h1):
                # x1 contains x1

                contains.append(1)

            else:
                contains.append(-1)

        contained_list = [item == -1 for item in contains]
        ext_rects = list(itertools.compress(ext_rects, contained_list))
        return ext_rects


def text_extraction(im_gray):
    contours, im_rgb, dilation = morphological_processing(im_gray, 30, 'kernel', 'dilationErosion')

    im_x, im_y = im_gray.shape
    im_area = im_x * im_y
    rects = []
    for c in contours:

        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h

        if 7 > h / w > 0.16 and 0.11 * im_area > rect_area > im_area * 0.00025:
            """
             if w and h aren't too small,
                area of the rectangle isn't too small or big respect the image area
                and ratio between h/w is bigger than 0.08 and smaller than 0.6
            we append the rectangle
             """
            rect = (x, y, w, h)
            rects.append(rect)

    no_overlapped_rects = erase_overlapped_regions(rects, im_x, im_y)
    detected_rects = get_max_rectangle(no_overlapped_rects)
    # for rectangle in no_overlapped_rects:
    x, y, w, h = detected_rects
    #    cv2.rectangle(im_rgb, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # List of rectangles to return the coordinates as required (tlx, tly, brx, bry)
    det_rect = (x, y, x + w, y + h)

    return det_rect


def detect_text_box(gray_im, plot_results):
    """
    Detect the bounding box of the text

    args:
    - gray_im: image in grayscale
    - plot_results: if true imshow of the orginal gray image with the bounding box contour superposed
    output:
    - rects: Coordinates, width and height
    """
    contours, im_rgb, dilation = morphological_processing(gray_im, 2, 'gaussian', 'dilation')

    im_x, im_y = gray_im.shape
    im_area = im_x * im_y
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h

        if 7 > h / w > 0.13 and w >= 0.02 * im_x and w * 0.50 > h and h >= 0.02 * im_y and 0.0020 * im_area < rect_area < 0.20 * im_area:
            """
             if w and h aren't too small,
                area of the rectangle isn't too small or big respect the image area
                and ratio between h/w is bigger than 0.08 and smaller than 0.6
            we append the rectangle
             """
            rect = (x, y, w, h)
            rects.append(rect)

    detected_rects = []
    if rects:
        """
        If we detect text rectangles
        """
        outside_rects = erase_inside_regions(rects)
        no_overlapped_rects = erase_overlapped_regions(outside_rects, im_x, im_y)

        list_dilation = []

        for rectangle in no_overlapped_rects:
            x, y, w, h = rectangle
            # List of rectangles to return the coordinates as required (tlx, tly, brx, bry)
            det_rect = (x, y, x + w, y + h)
            detected_rects.append(det_rect)
            list_dilation.append(sum(sum(dilation[y:(y + h), x:x + w])))
        # take the rectangle with biggest values inside
        max_dilation_rect = max(list_dilation)
        text_rect_1 = detected_rects[list_dilation.index(max_dilation_rect)]

        text_rect_2 = text_extraction(gray_im)
        text_rect = merge_text_detection_methods(text_rect_1, text_rect_2, im_x, im_y, dilation)
        cv2.rectangle(im_rgb, (text_rect[0], text_rect[1]), (text_rect[2], text_rect[3]), (255, 0, 0), 5)
    else:
        """
        Else if there are no rectangles detected in the image
        """
        text_rect = text_extraction(gray_im)

    if plot_results:
        fig = plt.figure(figsize=(11, 15))
        plt.imshow(im_rgb)
    else:
        pass
    return text_rect


def get_text_mask(im, bounding_box):
    "Receives an image WITH JUST ONE PAINTING and returns the mask with the pixels inside the text bbox to 0"
    tlx, tly, brx, bry = bounding_box
    mask = np.ones(shape=(im.shape[0], im.shape[1]),
                   dtype=np.uint8)

    mask[tly:bry, tlx:brx] = 0
    return mask


def get_text_mask_BGR(im):
    "Receives an image WITH JUST ONE PAINTING and returns the mask with the pixels inside the text bbox to 0"
    tlx, tly, brx, bry = detect_text_box(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), False)
    mask = np.ones(shape=(im.shape[0], im.shape[1]),
                   dtype=np.uint8)

    mask[tly:bry, tlx:brx] = 0
    return mask


def merge_text_detection_methods(m1_cords, m2_cords, im_w, im_h, dilation):
    try:
        x1, y1, xw1, yh1 = m1_cords
        x2, y2, xw2, yh2 = m2_cords
        possible_bbox1 = erase_vertical_rects(x1, y1, xw1 - x1, yh1 - y1)
        possible_bbox2 = erase_vertical_rects(x2, y2, xw2 - x2, yh2 - y2)
        rects = []
        rects.append((x1, y1, xw1 - x1, yh1 - y1))
        rects.append((x2, y2, xw2 - x2, yh2 - y2))

        if possible_bbox1 == False and possible_bbox2 == True:
            text_box = m2_cords
        elif possible_bbox2 == False and possible_bbox1 == True:
            text_box = m1_cords
        elif possible_bbox2 == True and possible_bbox1 == True:
            no_contained_rects = erase_inside_regions(rects)
            if len(no_contained_rects) > 1:
                no_overlapped = erase_overlapped_regions(no_contained_rects, im_w, im_h)
                if len(no_overlapped) > 1:
                    dil1 = dilation[y1:yh1, x1:xw1]
                    dil2 = dilation[y2:yh2, x2:xw2]
                    if sum(sum(dil1)) > sum(sum(dil2)):
                        text_box = m1_cords
                    elif sum(sum(dil1)) < sum(sum(dil2)):
                        text_box = m2_cords
                elif len(no_overlapped) == 1:
                    x, y, w, h = no_overlapped[0]
                    text_box = x, y, x + w, y + h
            elif len(no_contained_rects) == 1:
                x, y, w, h = no_contained_rects[0]
                text_box = x, y, x + w, y + h
        elif possible_bbox1 == False and possible_bbox2 == False:
            text_box = (0, 0, 0, 0)
        return text_box
    except:
        text_box = (0, 0, 0, 0)
        return text_box


def erase_vertical_rects(x, y, w, h):
    if w * 0.4 >= h:
        possible_bbox = True
    else:
        possible_bbox = False
    return possible_bbox
