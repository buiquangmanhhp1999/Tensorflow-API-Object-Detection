import cv2
from detector import Detector
from detector_faster import DetectorFaster
import time
import numpy as np
from detect_utils import get_args, align_image


arguments = get_args()
image_path = arguments.image_path

detector = Detector(path_config='./ssd_mobilenet_v2/pipeline.config', path_ckpt='./ssd_mobilenet_v2/ckpt/ckpt-23',
                    path_to_labels='./scripts/label_map.pbtxt')


image = cv2.imread(image_path)
start = time.time()
image, original_image, coordinate_dict = detector.predict(image)
align_image(original_image, coordinate_dict)
cv2.imshow('test', image)
cv2.waitKey(0)