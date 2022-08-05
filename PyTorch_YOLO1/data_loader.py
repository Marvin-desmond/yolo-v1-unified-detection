from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import tqdm

from typing import Tuple as tuple

import sys
sys.path.append("../")

import utils  # type: ignore
# from .. import utils # for autocompletion purposes

class VOCDatasetLoader(Dataset):
    def __init__(self, labels_file: str = "./VOC_2012_detections.parquet"):
            data = pd.read_parquet(labels_file)
            classes: Union[list[str], dict[str, int]] = data.detection_object.unique()
            classes = {_class: i for i, _class in enumerate(classes)}
            data.detection_object = data.detection_object.map(classes)
            self.classes = {i: _class for _class, i in classes.items()}
            self.data = data
            self.groups = self.data.groupby("path")
            self.paths = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = utils.read_image(f".{image_path}")
        image = utils.resize_dimensions(image, (448, 448))
        bboxes_classes = self.groups.get_group(image_path)[
            ['x_yolo', 'y_yolo', 'w_norm', 'h_norm', 'detection_object']
            ].values
        grid_detection = self.detection_to_grid(bboxes_classes)
        return image, grid_detection

    def detection_to_grid(self,
        bbox_class: tuple[float, float, float, float, int],
        S: int = 7, B: int = 2, C: int = 20) -> torch.Tensor:
        grid_detection = torch.zeros(S, S, C + 5 * B)
        for *bounding_box, detection_class in bbox_class:
            grid_x = int(np.floor(S * bounding_box[0]))
            grid_y = int(np.floor(S * bounding_box[1]))
            detection_class = int(detection_class)
            if not (grid_x == 7 or grid_y == 7):
                if grid_detection[grid_y, grid_x, 20] == 0:
                    # Set that there exists an object
                    grid_detection[grid_y, grid_x, 20] = 1
                    grid_detection[grid_y, grid_x, 21:25] = torch.FloatTensor(bounding_box)
                    grid_detection[grid_y, grid_x, detection_class] = 1
        return grid_detection



