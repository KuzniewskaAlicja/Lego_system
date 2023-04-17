import cv2
import numpy as np
from typing import Tuple
from collections import namedtuple
from copy import deepcopy
from operator import eq

from utils.io_handlers import objects_colors, images_names, description
from utils.color import create_mask


Information = namedtuple('Description', ['desc', 'circles'])


class ObjectDescriber:
    def __init__(self):
        self.obj_keys = objects_colors

    def __call__(self, obj: np.ndarray):
        return Information(desc=self._counting_objects(obj),
                           circles=self._circles_nb(obj))

    def _counting_objects(self, img: np.ndarray) -> dict:
        counter = 0
        BLOCK_HEIGHT = 43.0
        objects_desc = {}
        for color in self.obj_keys:
            mask = create_mask(img, color)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 700:
                    height = min(cv2.minAreaRect(cnt)[1])
                    counter += height // BLOCK_HEIGHT
            objects_desc[color] = str(int(counter))
            counter = 0

        return objects_desc
    
    def _circles_nb(self, img: np.ndarray) -> int:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        object_circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 34,
                                          param1=39, param2=20, minRadius=6,
                                          maxRadius=17)
        object_circles = np.uint16(np.around(object_circles))
        
        return len(object_circles[0,:])


class DescMatcher:
    images_description = deepcopy(description)

    def __init__(self, img_nb: int):
        self.img_desc = DescMatcher.images_description[images_names[img_nb]]
        self.img_name = images_names[img_nb]

    def __call__(self, obj_desc: dict) -> Tuple[str, int]:
        scores = [self._calc_score(obj_desc.desc, file_desc)
                  for file_desc in self.img_desc]

        max_score_idx = np.argmax(scores)
        best_matching_desc = self.img_desc[max_score_idx]
        del DescMatcher.images_description[self.img_name][max_score_idx]

        return (self.img_name,
                description[self.img_name].index(best_matching_desc))
    
    def _calc_score(self, *descriptions) -> int:
        descriptions = list(map(lambda item: item.values(), descriptions))
        return sum([int(eq(*obj_nb))
                    for obj_nb in zip(*descriptions)])
