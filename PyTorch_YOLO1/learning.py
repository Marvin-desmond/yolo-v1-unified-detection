from data_loader import VOCDatasetLoader
import tqdm
import numpy as np
import cv2

from typing import Tuple as tuple

# from .. import utils # for autocompletion
import sys
sys.path.append("../")
import utils  # type: ignore
import transforms

dataset = VOCDatasetLoader()
classes, data = dataset.classes, dataset.data


class YOLOConfig:
    IMG_SIZE: tuple[int, int] = (448, 448)  # for YOLO v1
    S: int = 7
    B: int = 2
    C: int = len(classes)


sample_data = [{
    "path": f".{path}",
    "bboxes": group[['xmin', 'ymin', 'xmax', 'ymax', 'detection_object']].values
} for path, group in tqdm.tqdm(data.groupby("path"))]

index = 0
for sample in sample_data:
    image = utils.read_image(sample['path'])
    bboxes = [
        [*utils.resize_bounding_boxes(
            image.shape[:2],i, YOLOConfig.IMG_SIZE
            ), j]
        for *i, j in sample['bboxes']
        ]
    image = utils.visualize_bounding_boxes(
        utils.resize_dimensions(image,YOLOConfig.IMG_SIZE),
        bboxes, classes)
    image = utils.visualize_center_points(image, [box[:4] for box in bboxes])
    image = utils.draw_grid(image)
    utils.show_image(image)
    index += 1
    if index == 3:
        break
    # image = utils.read_image(f'.{sample[0]}')
    # utils.show_image(image)
