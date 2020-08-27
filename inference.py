import cv2
from detector import Detector
from detector_faster import DetectorFaster
import argparse
import time
import numpy as np


def get_center_point(box):
    xmin, ymin, xmax, ymax = box
    return (xmin + xmax) // 2, (ymin + ymax) // 2


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./test2.jpg')
    return arg.parse_args()


def find_missed_corner(coord_li):
    di = dict()
    status = np.array([0, 0, 0, 0])

    def get_x(element):
        return element[0]

    def get_y(element):
        return element[1]

    sort_x = sorted(coord_li, key=get_x)
    sort_y = sorted(coord_li, key=get_y)
    check_xmin = sort_x[0][0]
    check_ymin = sort_y[0][1]
    check_xmax = sort_x[2][0]
    check_ymax = sort_y[2][1]
    threshold_x = check_xmax - check_xmin + 10
    threshold_y = check_ymax - check_ymin + 10

    """
    (xmin, ymin) ==> top_left
    (xmin, ymax) ==> bottom_left
    (xmax, ymin) ==> top_right
    (xmax, ymax) ==> top_left
    """
    for coordinate in coord_li:
        x, y = coordinate
        if threshold_x > x - check_xmin >= 0:
            if threshold_y > y - check_ymin >= 0:
                di['top_left'] = (x, y)
                status[0] = 1
            else:
                di['bottom_left'] = (x, y)
                status[2] = 1
        else:
            if threshold_y > y - check_ymin >= 0:
                di['top_right'] = (x, y)
                status[1] = 1
            else:
                di['bottom_right'] = (x, y)
                status[3] = 1

    # calculate missed corner coordinate
    # case 1: missed corner is "top_left"
    index = np.argmin(status)
    if index == 0:
        midpoint = int(np.add(di['top_right'], di['bottom_left'])/2)
        y = 2 * midpoint[1] - di['bottom_right'][1]
        x = 2 * midpoint[0] - di['bottom_right'][0]
        di['top_left'] = (x, y)
    elif index == 1:    # "top_right"
        midpoint = int(np.add(di['top_left'], di['bottom_right']) / 2)
        y = 2 * midpoint[1] - di['top_left'][1]
        x = 2 * midpoint[0] - di['bottom_left'][0]
        di['top_right'] = (x, y)
    elif index == 2:    # "bottom_left"
        midpoint = int(np.add(di['top_left'], di['bottom_right']) / 2)
        y = 2 * midpoint[1] - di['top_right'][1]
        x = 2 * midpoint[0] - di['top_right'][0]
        di['bottom_left'] = (x, y)
    else:               # "bottom_right"
        midpoint = int(np.add(di['bottom_left'], di['top_right']) / 2)
        y = 2 * midpoint[1] - di['top_left'][1]
        x = 2 * midpoint[0] - di['top_left'][0]
        di['bottom_right'] = (x, y)
    return di


def align_image(coord_list):
    if len(coord_list) < 3:
        raise ValueError('Please try again')

    final_points = list(map(get_center_point, coord_list))

    # find missed corner
    if len(coord_list) == 3:
        coordinate_dict = find_missed_corner(final_points)


arguments = get_args()
image_path = arguments.image_path

detector = Detector(path_config='./ssd_mobilenet_v2/pipeline.config', path_ckpt='./ssd_mobilenet_v2/ckpt/ckpt-13',
                    path_to_labels='./scripts/label_map.pbtxt')


image = cv2.imread(image_path)
start = time.time()
image, original_image, coordinate_list = detector.predict(image)
align_image(coordinate_list)
# cv2.imshow('test', image)
# cv2.waitKey(0)