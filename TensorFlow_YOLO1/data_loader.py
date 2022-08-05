from typing import Union
import numpy as np
import tensorflow as tf
import pandas as pd
import tqdm

from typing import (
    List as list, 
    Tuple as tuple, 
    Dict as dict
)

import sys
sys.path.append("../")

import utils  # type: ignore
# from .. import utils # for autocompletion purposes

def load_data(path: str) -> tuple[
    Union[list[str], dict[str, int]],
    pd.DataFrame
    ]:
    data = pd.read_parquet(path)
    classes: Union[list[str], dict[str, int]] = data.detection_object.unique()
    classes = {_class: i for i, _class in enumerate(classes)}
    data.detection_object = data.detection_object.map(classes)
    classes = {i: _class for _class, i in classes.items()}
    return classes, data


def create_tf_datasets(
        paths: list[str],
        bboxes_classes: list[tuple[float, float, float, float, int]]
        ) -> tf.data.Dataset:
    paths_tf = tf.data.Dataset.from_tensor_slices(paths)
    bboxes_classes_tf = tf.data.Dataset.from_tensor_slices(bboxes_classes)
    return tf.data.Dataset.zip(
        (paths_tf, bboxes_classes_tf)
    )


def path_to_image(path: str) -> np.uint8:
    image = utils.read_image(f".{path.numpy().decode('utf-8')}")
    image = utils.resize_dimensions(image, (448, 448))
    return image


def detection_to_grid(
        bbox_class: tuple[float, float, float, float, int],
        S: int = 7, B: int = 2, C: int = 20) -> np.ndarray:
    grid_detection = np.zeros((S, S, C + 5 * B))
    ##################################################################
    #                                                                #
    #[20 grids ~~ classes, 2 bounding boxes [x, y, w, h, confidence]]#
    #                                                                #
    ##################################################################
    for *bounding_box, detection_class in bbox_class.numpy():
        detection_class = detection_class.astype(np.uint8)
        grid_x = int(np.floor(S * bounding_box[0]))
        grid_y = int(np.floor(S * bounding_box[1]))
        if not (grid_x == 7 or grid_y == 7):
            if grid_detection[grid_y, grid_x, 20] == 0:
                # Set that there exists an object
                grid_detection[grid_y, grid_x, 20] = 1
                grid_detection[grid_y, grid_x, 21:25] = bounding_box
                grid_detection[grid_y, grid_x, detection_class] = 1 
    return grid_detection

def VOCDataLoader() -> tf.data.Dataset:
    data_path = "VOC_2012_detections.parquet"
    classes, data = load_data(data_path)
    paths = []
    bboxes_classes = []
    for path, group in tqdm.tqdm(data.groupby("path")):
        paths.append(path)
        bboxes_classes.append(group[['x_yolo', 'y_yolo', 'w_norm', 'h_norm', 'detection_object']].values)
    paths = tf.constant(paths)
    bboxes_classes = tf.ragged.constant(bboxes_classes)

    dataset = create_tf_datasets(paths, bboxes_classes)
    dataset = dataset.map(
        lambda path, bbox_class: (
            tf.py_function(
                func=path_to_image,
                inp=[path],
                Tout=tf.float32
            ), bbox_class)
    ).map(
        lambda image, bbox_class: (image,
            tf.py_function(
                func=detection_to_grid,
                inp=[bbox_class],
                # adding square brackets to Tout type adds dimension # DON'T
                Tout=tf.float32 
            ))
    ).batch(16)

    return dataset


if __name__ == "__main__":
    data_path = "VOC_2012_detections.parquet"
    classes, data = load_data(data_path)
    paths = data[['path']].values
    bboxes = data[['x_yolo', 'y_yolo', 'w_norm', 'h_norm']].values
    detection_classes = data[['detection_object']].values
    for index, (name, group) in enumerate(data.groupby("path")):
            image = utils.visualize_bounding_boxes(
                utils.read_image(f".{name}"),   
                group[['xmin', 'ymin', 'xmax', 'ymax', 'detection_object']].values,
                classes
            )
            image = utils.show_image(image)
            if index == 10:
                break
