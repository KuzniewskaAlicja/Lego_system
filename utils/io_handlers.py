from json import load, dump
from typing import Tuple
from copy import deepcopy
import os

from utils import config, DirectoryNotFoundError, FileNotFoundError


class InputInfo:
    def __init__(self):
        pass

    def __call__(self, file_path: str) -> Tuple[dict, Tuple[str, ...],
                                                Tuple[str, ...]]:
        if not os.path.exists(os.path.dirname(file_path)):
            raise DirectoryNotFoundError(
                f"Directory {os.path.abspath(os.path.dirname(file_path))} "
                "does not exist")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Cannot find provided file path {os.path.abspath(file_path)}")

        with open(file_path, 'r') as file:
            description = load(file)

        image_names = tuple(description.keys())
        object_colors = tuple(tuple(description.values())[0][0].keys())

        return description, image_names, object_colors


description, images_names, objects_colors = InputInfo()(config["description_file"])


class OutputFile:
    def __init__(self):
        self.path = config['output_file']
        if not os.path.exists(os.path.dirname(self.path)):
            raise DirectoryNotFoundError(
                f"Directory {os.path.abspath(os.path.dirname(self.path))} "
                "does not exist")

        self.objects_description = deepcopy(description)
    
    def insert(self, image_name: str, index: int, circles: int):
        self.objects_description[image_name][index] = circles

    def save(self):
        with open(self.path, 'w') as file:
            dump(self.objects_description, file, indent=4)
