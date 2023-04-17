import cv2
import numpy as np
import glob
import os

from utils import config, DirectoryNotFoundError, NoImagesError


class ObjectsLoader:
    def __init__(self):
        pass

    def __call__(self) -> list:
        path = config['images_dir']
        if not os.path.exists(path):
            raise DirectoryNotFoundError(f"Directory {os.path.abspath(path)} "
                                         "does not exist")

        images = self._load_imgs(config['images_dir'])
        if not images:
            raise NoImagesError("There are no images in provided "
                                f"directory {os.path.abspath(path)}")

        return [self._img_objects(img) for img in images]
 
    def _load_imgs(self, images_dir: str) -> list:
        resize = lambda img, factor: cv2.resize(img, None,
                                                fx=factor,
                                                fy=factor)
        images = [resize(cv2.imread(path), 0.5)
                  for path in sorted(glob.glob(images_dir + '/*.jpg'))]

        return images

    def _create_object_image(self, rect: tuple, box: np.ndarray, 
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

    def _img_objects(self, img: np.ndarray) -> list:
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
                objects.append(self._create_object_image(min_rect,
                                                         box_contour,
                                                         img))

        return objects
