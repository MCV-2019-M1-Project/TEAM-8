import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from text_detector import detect_text_box


def my_find_peaks(array, height_ratio, distance_between_peaks):
    """
    Peak finding algorithm in which we can select the required height of the
    returned peaks with respect to the highest peak in the array (via height_ratio),
    and the required distance between peaks.
    """
    peaks, _ = find_peaks(array,
                          height=np.max(array) / height_ratio,
                          distance=distance_between_peaks)
    return peaks


def remove_single_background(im, height_ratio=10, distance_between_peaks=10):
    """
        Removes the background of an image assuming it contains a single painting

        Arguments:
            - im: a numpy image, in the BGR color space
            - height_ratio, distance_between_peaks: they control the peak finding algo

        Returns:
            - (cropped_image, binary_mask, [left,top,right,bottom])
    """

    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    col_var = np.abs(np.gradient(gray_image.mean(0)))

    left_cd = my_find_peaks(col_var[:len(col_var) // 2], height_ratio,
                            distance_between_peaks)
    left = left_cd[0]

    right_cd = my_find_peaks(col_var[len(col_var) // 2:], height_ratio,
                             distance_between_peaks) + len(col_var) // 2
    right = right_cd[-1]

    row_var = np.abs(np.gradient(gray_image.mean(1)))

    top_cd = my_find_peaks(row_var[:len(row_var) // 2], height_ratio,
                           distance_between_peaks)
    top = top_cd[0]

    bottom_cd = my_find_peaks(row_var[len(row_var) // 2:], height_ratio,
                              distance_between_peaks) + len(row_var) // 2
    bottom = bottom_cd[-1]

    mask = np.zeros(shape=(im.shape[0], im.shape[1]),
                    dtype=np.uint8)

    mask[top:bottom, left:right] = 255

    cropped_image = im[top:bottom, left:right].copy()

    return cropped_image, mask, [left, top, right, bottom]


def find_silences(array, length=50, threshold=0.03):
    def is_silence(sub_array):
        return np.std(sub_array) <= threshold

    silences = []
    current_silence = [0, 0, 0]
    in_silence = False
    for sub_array_start in range(len(array))[::length]:
        if is_silence(array[sub_array_start:sub_array_start + length]):
            if not in_silence:
                current_silence = [sub_array_start, sub_array_start + length,
                                   np.std(array[sub_array_start:sub_array_start + length])]
                in_silence = True
            else:
                current_silence[1] = sub_array_start + length
        else:
            if in_silence:
                silences.append(current_silence[:])
            in_silence = False
    return silences


def get_partition_horizontal(image):
    """
    Receives the path of an image that can contain 1 or 2 paintings horizontally and
    Returns:
        - x_coordinate: int that represents the column of
                        the image that parts the image if
                        it thinks there are 2 paintings.
        OR

        - None: if it thinks there is only one painting.
    """

    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    col_var = gaussian_filter1d(np.abs(np.gradient(im.mean(0))), 6)
    third_index = int(len(col_var) * 0.20)
    left_cd = my_find_peaks(col_var, 2, 150)
    left_cd = [x for x in left_cd if third_index < x < len(col_var) - third_index]
    if len(left_cd) < 1:
        return None
    silences = find_silences(col_var, length=50, threshold=0.03)
    silences = [x for x in silences if x[0] > third_index and x[1] < len(col_var) - third_index]
    if not silences:
        return None
    candidate_to_silence = None
    if len(left_cd) == 2:
        candidates_to_silence = [x for x in silences if x[0] > left_cd[0] and x[1] < left_cd[1]]
        if candidates_to_silence:
            candidate_to_silence = candidates_to_silence[0]
    if candidate_to_silence is None:
        candidate_to_silence = sorted(silences, key=lambda x: x[1] - x[0], reverse=True)[0]
    cd = (candidate_to_silence[0] + candidate_to_silence[1]) // 2
    return cd


def remove_background(image, height_ratio=10, distance_between_peaks=10, single_sure=False):
    """
    Receives an image and returns a list with the cropped images that are found on it

    Arguments:
        - image : an opencv image
    Returns:
        list_of_paintings: list of cropped images found in the image, in order.
                           Left to right or top to bottom (currently not supported).
    """

    middle_h = None if single_sure else get_partition_horizontal(image)
    if middle_h is None:
        return [remove_single_background(image, height_ratio, distance_between_peaks)[0]]
    else:
        return [
            remove_single_background(image[:, :middle_h, :],
                                     height_ratio, distance_between_peaks)[0],
            remove_single_background(image[:, middle_h:, :],
                                     height_ratio, distance_between_peaks)[0]
        ]


def get_mask(image, height_ratio=10, distance_between_peaks=10, single_sure=False):
    """
    Receives an image and returns the background mask

    Arguments:
        - image : an opencv image
        - single_sure : Set it to True if you know beforehand that will only enter images
                      with a single painting.
    Returns:
        binary_mask: background binary mask

    """

    middle_h = None if single_sure else get_partition_horizontal(image)
    if middle_h is None:
        return remove_single_background(image, height_ratio, distance_between_peaks)[1]
    else:
        return np.concatenate((
            remove_single_background(image[:, :middle_h, :],
                                     height_ratio, distance_between_peaks)[1],

            remove_single_background(image[:, middle_h:, :],
                                     height_ratio, distance_between_peaks)[1]),
                axis=1)


def get_bbox(image, height_ratio=10, distance_between_peaks=10, single_sure=False):
    """
    Receives an image and returns the background mask bounding_box

    Arguments:
        - image : an opencv image
        - single_sure : Set it to True if you know beforehand that will only enter images
                      with a single painting.
    Returns:
        - List with the bounding boxes of all the paintings found in the image

    """

    middle_h = None if single_sure else get_partition_horizontal(image)
    if middle_h is None:
        return [remove_single_background(image, height_ratio, distance_between_peaks)[2]]
    else:

        first_painting_bbox = remove_single_background(image[:, :middle_h, :],
                                                       height_ratio, distance_between_peaks)[2]

        [left, top, right, bottom] = remove_single_background(image[:, middle_h:, :],
                                                              height_ratio, distance_between_peaks)[2]

        adjusted_second_painting_bbox = [left + middle_h, top, right + middle_h, bottom]

        return [first_painting_bbox, adjusted_second_painting_bbox]


def get_background_and_text_mask(im, single_sure=False, return_only_bounding_box_pos=False):
    """
    Args
        - im: Assuming an image in bgr color space
        - single_sure: Set to True if you know the image will only contain one painting
        - return_only_bounding_box_pos: Pues eso
    """

    ims = remove_background(im, single_sure=single_sure)
    mask = get_mask(im, single_sure=single_sure)
    bg_bboxes = get_bbox(im, single_sure=single_sure)

    text_bboxes = []
    for i in range(len(ims)):
        tlx, tly, brx, bry = detect_text_box(cv2.cvtColor(ims[i], cv2.COLOR_RGB2GRAY), False)
        tlx_bg, tly_bg, brx_bg, bry_bg = bg_bboxes[i]

        mask[tly + tly_bg:bry + tly_bg, tlx + tlx_bg:brx + tlx_bg] = 0
        text_bboxes.append([tlx + tlx_bg, tly + tly_bg, brx + tlx_bg, bry + tly_bg])

    if return_only_bounding_box_pos:
        return text_bboxes
    else:
        return mask, text_bboxes


if __name__ == "__main__":
    pass