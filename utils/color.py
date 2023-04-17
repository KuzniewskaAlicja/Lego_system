from collections import namedtuple
from typing import Tuple
import numpy as np
import cv2


ColorRange = namedtuple('ColorRange', ['low', 'high'])
Operation = namedtuple('Operation', ['morph_op', 'kernel', 'iter'])


class Color:
    possible_operations = {'dilation': cv2.MORPH_DILATE,
                           'erosion': cv2.MORPH_ERODE,
                           'open': cv2.MORPH_OPEN}

    def __init__(self, ranges: Tuple[np.ndarray, np.ndarray],
                 operations: dict, name: str):
        self.name = name
        self.ranges = [ColorRange(*range_) for range_ in ranges]
        self.operations = [self._get_operation(name, params)
                           for name, params in operations.items()]

    def color_mask(self, mask: np.ndarray,
                   hsv_image: np.ndarray) -> np.ndarray:
        for range_ in self.ranges:
            mask = cv2.add(mask, cv2.inRange(hsv_image,
                                             range_.low,
                                             range_.high))
        
        for operation in self.operations:
            mask = cv2.morphologyEx(mask,
                                    operation.morph_op,
                                    operation.kernel,
                                    iterations=operation.iter)
        
        return mask

    def _get_operation(self, operation_name: str, params: dict) -> Operation:
        return Operation(morph_op=Color.possible_operations[operation_name],
                         kernel=np.ones(params['kernel_size'],
                                        dtype=np.uint8),
                         iter=params['iter'])


colors = [Color(ranges=(np.array([[0, 54, 100], [10, 255,255]]),
                        np.array([[160, 50, 50], [180, 200,200]])),
                operations={'dilation': {'kernel_size': (3, 3), 'iter': 1}},
                name='red'),
          Color(ranges=(np.array([[104, 70, 70], [130, 255, 255]]),
                        np.array([[104, 24, 180], [124, 64, 220]])),
                operations={'dilation': {'kernel_size': (3, 3), 'iter': 5}},
                name='blue'),
          Color(ranges=(np.array([[13, 109, 160], [35, 255, 240]]),
                        np.array([[14, 32, 210], [35, 87, 250]])),
                operations={'dilation': {'kernel_size': (3, 3), 'iter': 5}},
                name='yellow'),
          Color(ranges=(np.array([[37, 7, 56], [110, 60, 140]]),
                        np.array([[37, 3, 56], [110, 60, 180]])),
                operations={'open': {'kernel_size': (3, 3), 'iter': 2}},
                name='grey'),
          Color(ranges=(np.array([[37, 8, 177], [62, 50, 255]]),),
                operations={'open': {'kernel_size': (2, 2), 'iter': 1},
                            'dilation': {'kernel_size': (3, 3), 'iter': 4}},
                name='white')]


def create_mask(img: np.ndarray, color_name: str):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color = next((color for color in colors
                  if color.name == color_name))

    return color.color_mask(mask, hsv_image)
