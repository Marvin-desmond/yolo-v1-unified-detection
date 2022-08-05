import cv2
import numpy as np
from PIL import Image
import time 
from transforms import (
    b_minmax_to_b_center, 
    b_center_to_b_normalized, 
    b_normalized_to_b_yolo) # type: ignore

import random 

from typing import List as list, Tuple as tuple

def read_image(path: str) -> np.uint8:
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def stack_images(image_one: np.uint8, image_two: np.uint8) -> np.ndarray:
    return np.hstack((image_one, image_two))


def numpy_to_pil(numpy_img: np.uint8) -> Image:
    return Image.fromarray(numpy_img)


def pil_to_numpy(pil_img: Image) -> np.ndarray:
    return np.asarray(pil_img)


def show_image(image: np.uint8, title: str = "Image") -> None:
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def xxyy_to_xywh(bounding_boxes: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bounding_boxes
    return xmin, ymin, xmax - xmin, ymax - ymin


def resize_bounding_boxes(
    old_size: tuple[int, int],
    bounding_boxes: tuple[int, int, int, int],
    new_size: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = old_size
    height_ratio, width_ratio = new_size[0] / height, new_size[1] / width
    x, y, w, h = bounding_boxes
    x, w = int(width_ratio * x), int(width_ratio * w)
    y, h = int(height_ratio * y), int(height_ratio * h)
    return x, y, w, h


def resize_scale_factor(image: np.uint8, fx=0.5, fy=0.5) -> np.uint8:
    return cv2.resize(image, None, fx=fx, fy=fy)


def resize_dimensions(image: np.uint8, dimensions: tuple[int, int] = (512, 512)) -> np.uint8:
    return cv2.resize(image, dimensions)

########################################
#    VISUAL UTILS

def visualize_bounding_boxes(
    image: np.uint8,
    bboxes_classes: list[tuple[int, int, int, int, int]],
    classes: list[str]) -> np.uint8:
    for *bbox, _class in bboxes_classes:
        xmin, ymin, xmax, ymax = bbox
        image = cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 0, 255),
            1
        )
        image = cv2.putText(image, classes[_class],
                    (xmin + 10, ymin + 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.9,
                    (0, 0, 255),
                    2)
    return image

coolors: list[tuple[int, int, int]] = [
    (205, 180, 219),
    (255, 200, 221),
    (255, 175, 204),
    (189, 224, 254),
    (162, 210, 255)
]

def visualize_center_points(
    image: np.uint8, 
    bboxes: list[tuple[int, int ,int, int]],
) -> np.uint8:
    bboxes = [b_minmax_to_b_center(box) for box in bboxes]
    for box in bboxes:
        coolor = random.choice(coolors)
        x, y, w, h = box
        # norm_x, norm_y, *_ = b_center_to_b_normalized(image.shape[:2], box)
        # grid_x, grid_y = np.floor(norm_x * 7), np.floor(norm_y * 7)
        # grid_x, grid_y = int(grid_x * 64), int(grid_y * 64)
        # image = cv2.circle(image, (x, y), 10, coolor, thickness=-1)
        # x_, y_ = int(x - w / 2), int(y - h / 2)
        # _x, _y = int(x + w / 2), int(y + h / 2)
        # image = cv2.line(image, (x_, y), (_x, y), (255, 255, 255), 2)
        # image = cv2.line(image, (x, y_), (x, _y), (255, 255, 255), 2)
        # image = cv2.line(image, (grid_x, y), (x, y), (0, 255, 0), 2)
        # image = cv2.line(image, (x, grid_y), (x, y), (0, 255, 0), 2)
        # image = cv2.circle(image, (grid_x, grid_y), 10, coolor, thickness=-1)

        ##################
        #                #
        #       OR       #
        #                #
        ##################

        # GETTING YOLO FORMAT
        norm_x, norm_y, norm_w, norm_h = b_center_to_b_normalized(image.shape[:2], box)
        yolo_x, yolo_y, norm_w, norm_h = b_normalized_to_b_yolo([norm_x, norm_y, norm_w, norm_h])
        grid_x, grid_y = np.floor(norm_x * 7), np.floor(norm_y * 7)
        grid_x, grid_y = int(grid_x * 64), int(grid_y * 64)

        # DERIVING VISUAL COORDINATES
        x, y = grid_x + int(yolo_x * 64), grid_y + int(yolo_y * 64)
        w, h = int(norm_w * 448), int(norm_h * 448)


        image = cv2.circle(image, (x, y), 10, coolor, thickness=-1)
        x_, y_ = int(x - w / 2), int(y - h / 2)
        _x, _y = int(x + w / 2), int(y + h / 2)
        image = cv2.line(image, (x_, y), (_x, y), (255, 255, 255), 2)
        image = cv2.line(image, (x, y_), (x, _y), (255, 255, 255), 2)
        image = cv2.line(image, (grid_x, y), (x, y), (0, 255, 0), 2)
        image = cv2.line(image, (x, grid_y), (x, y), (0, 255, 0), 2)
        image = cv2.circle(image, (grid_x, grid_y), 10, coolor, thickness=-1)
        
    return image

def show_sleep(image):
    cv2.imshow("image", image)
    cv2.waitKey(1)
    time.sleep(0.2)

def draw_grid(image: str, S: int = 7) -> np.uint8:
    height, width = image.shape[:2]
    scale_y, scale_x = height / S,  width / S
    index = 0
    for y_ in range(S):
        image = cv2.line(image, (int(y_ * scale_y), 0), (int(y_ * scale_y), width), (0, 255, 0), 1)
        # show_sleep(image)
    for x_ in range(S):
        image = cv2.line(image, (0, int(x_ * scale_x)), (height, int(x_ * scale_x)), (0, 255, 0), 1)        
        # show_sleep(image)
    return image 


