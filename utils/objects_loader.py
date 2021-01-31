import cv2
from json import load
import numpy as np
from pathlib import Path

class ObjectsLoader:
    def __init__(self, imgs_dir, json_path):
        self.imgs_dir = Path(imgs_dir)
        self.json_path = json_path
        self.__load_imgs()
        self.__load_json_content()
    
    def __load_imgs(self):
        self.imgs = [ObjectsLoader.resize(cv2.imread(str(path)), 0.5)
                     for path in sorted(self.imgs_dir.iterdir())]
    
    def __load_json_content(self):
        self.json_content = load(open(self.json_path, 'r'))
    
    def img_object_keys(self):
        img_keys = tuple(self.json_content.keys())
        return (img_keys,
                tuple(self.json_content[img_keys[0]][0].keys()))

    def get_objects(self):
        return [ObjectsLoader.img_objects(img) for img in self.imgs]

    @staticmethod
    def resize(img: np.ndarray, factor: float) -> np.ndarray:
        return cv2.resize(img, None, fx=factor, fy=factor)

    @staticmethod
    def create_object_image(rect: tuple, box: np.ndarray, 
                            img: np.ndarray) -> np.ndarray:
        w, h = tuple(map(int, rect[1]))
        box_pts = box.astype('float32')
        image_pts = np.float32([[0, h - 1],
                                [0, 0],
                                [w - 1, 0],
                                [w - 1, h - 1]])
        mat = cv2.getPerspectiveTransform(box_pts, image_pts)
        box_img = cv2.warpPerspective(img, mat, (w, h))

        return box_img

    @staticmethod
    def img_objects(img):
        objects = []
        edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 111, 46)
        edges = cv2.dilate(edges, (3,3), iterations=35)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 2000:
                min_rect = cv2.minAreaRect(cnt)
                box_contour = cv2.boxPoints(min_rect)
                box_contour = np.int0(box_contour)
                objects.append(ObjectsLoader.create_object_image(min_rect, 
                                                                 box_contour,
                                                                 img))
        
        return objects
