import numpy as np 
import tensorflow as tf

import sys
sys.path.append("../")
import transforms 

# Intersection over Union
def intersection_over_union(
    predicted_boxes: tf.constant, 
    ground_boxes: tf.constant,
    bbox_format = "midpoint",
    ):
    if bbox_format == "midpoint":
        predicted_x_min = predicted_boxes[..., 0:1] - predicted_boxes[..., 2:3] / 2
        predicted_y_min = predicted_boxes[..., 1:2] - predicted_boxes[..., 3:4] / 2
        predicted_x_max = predicted_boxes[..., 0:1] + predicted_boxes[..., 2:3] / 2
        predicted_y_max = predicted_boxes[..., 1:2] + predicted_boxes[..., 3:4] / 2
        ground_x_min = ground_boxes[..., 0:1] - ground_boxes[..., 2:3] / 2
        ground_y_min = ground_boxes[..., 1:2] - ground_boxes[..., 3:4] / 2
        ground_x_max = ground_boxes[..., 0:1] + ground_boxes[..., 2:3] / 2
        ground_x_max = ground_boxes[..., 1:2] + ground_boxes[..., 3:4] / 2
    elif bbox_format == "corners":
        predicted_x_min = predicted_boxes[..., 0:1]
        predicted_y_min = predicted_boxes[..., 1:2]
        predicted_x_max = predicted_boxes[..., 2:3]
        predicted_y_max = predicted_boxes[..., 3:4]
        ground_x_min = ground_boxes[..., 0:1]
        ground_y_min = ground_boxes[..., 1:2]
        ground_x_max = ground_boxes[..., 2:3]
        ground_y_max = ground_boxes[..., 3:4]

    predicted_height = predicted_y_max - predicted_y_min
    predicted_width = predicted_x_max - predicted_x_min 
    predicted_area = tf.abs(predicted_height * predicted_width)

    ground_height = ground_y_max - ground_y_min
    ground_width = ground_x_max - ground_x_min 
    ground_area = tf.abs(ground_height * ground_width)

    intersection_x_min = tf.maximum(predicted_x_min, ground_x_min)
    intersection_y_min = tf.maximum(predicted_y_min, ground_y_min)
    intersection_x_max = tf.minimum(predicted_x_max, ground_x_max)
    intersection_y_max = tf.minimum(predicted_y_max, ground_y_max)

    intersection_width = tf.maximum(intersection_x_max - intersection_x_min, 0)
    intersection_height = tf.maximum(intersection_y_max - intersection_y_min, 0)

    intersection = intersection_height * intersection_width
    union = predicted_area + ground_area - intersection + tf.constant(1e6)
    jaccard_index = intersection / union
    return jaccard_index


# Highest Confidence Box
def highest_confidence_box(
    predictions: tf.Tensor, S: int = 7):
    """
    Input shape : [Batch Size, S, S, (C + B * 5)] => [16, 7, 7, 30]
    Straight from YOLO output
    to Normalized [CLASS, CONF, X, Y, W, H] w.r.t (0, 1]
    """
    batch_size = predictions.shape[0]
    grid_classes = predictions[..., :20]
    confidence_one = predictions[..., 20:21]
    boxes_one = predictions[..., 21:25]
    confidence_two = predictions[..., 25:26]
    boxes_two = predictions[..., 26:30]
    confidence_scores = tf.concat([confidence_one, confidence_two], axis = -1)

    best_scores = tf.expand_dims(tf.math.argmax(confidence_scores, axis = -1), axis = -1)
    best_scores = tf.cast(best_scores, tf.float32)
    best_boxes = boxes_one * (1 - best_scores) + best_scores * boxes_two

    cell_indices = tf.reshape(
    tf.tile(
        tf.constant(tf.expand_dims(tf.range(S), 0)),
        tf.constant([batch_size, S], tf.int32)
    ),
    [batch_size, S, S, 1])
    cell_indices = tf.cast(cell_indices, tf.float32)
    0.25 + 5
    norm_x = 1 / S * (best_boxes[..., :1] + cell_indices)
    norm_y = 1 / S * (best_boxes[..., 1:2] + tf.transpose(cell_indices, (0, 2, 1, 3)))
    norm_w_h = 1 / S * best_boxes[..., 2:4]

    norm_boxes = tf.concat([norm_x, norm_y, norm_w_h], axis = -1)
    best_confidences = tf.expand_dims(tf.reduce_max(confidence_scores, axis = -1), axis = -1)
    best_classes = tf.expand_dims(tf.math.argmax(grid_classes, axis = -1), axis = -1)
    best_classes = tf.cast(best_classes, tf.float32)
    norm_predictions = tf.concat([best_classes, best_confidences, norm_boxes], axis = -1)
    return norm_predictions

# Mean Average Precision
