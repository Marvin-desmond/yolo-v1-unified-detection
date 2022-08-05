import numpy as np 
import torch
import transforms 
from collections import Counter

# Intersection over Union
def intersection_over_union(predicted_boxes, ground_boxes, bbox_format):
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
    predicted_area = torch.abs(predicted_height * predicted_width)

    ground_height = ground_y_max - ground_y_min
    ground_width = ground_x_max - ground_x_min 
    ground_area = torch.abs(ground_height * ground_width)

    intersection_x_min = torch.max(predicted_x_min, ground_x_min)
    intersection_y_min = torch.max(predicted_y_min, ground_y_min)
    intersection_x_max = torch.min(predicted_x_max, ground_x_max)
    intersection_y_max = torch.min(predicted_y_max, ground_y_max)

    intersection_width, _ = torch.max(intersection_x_max - intersection_x_min, 0)
    intersection_height, _ = torch.max(intersection_y_max - intersection_y_min, 0)

    intersection = intersection_height * intersection_width
    union = predicted_area + ground_area - intersection + 1e6
    jaccard_index = intersection / union
    return jaccard_index

# Highest Confidence Box
def highest_confidence_box(
    predictions: torch.Tensor, S: int = 7):
    """
    Input shape : [Batch Size, S, S, (C + B * 5)] => [16, 7, 7, 30]
    Straight from YOLO output
    to Normalized [X, Y, W, H, CONF, CLASS] w.r.t (0, 1]
    """
    batch_size = predictions.shape[0]
    grid_classes = predictions[..., :20]
    confidence_one = predictions[..., 20:21]
    boxes_one = predictions[..., 21:25]
    confidence_two = predictions[..., 25:26]
    boxes_two = predictions[..., 26:30]
    confidence_scores = torch.cat([confidence_one, confidence_two], dim = -1)

    best_scores = torch.argmax(confidence_scores, dim = -1).unsqueeze(-1)
    best_scores = best_scores.float()
    best_boxes = boxes_one * (1 - best_scores) + best_scores * boxes_two

    cell_indices = torch.reshape(
    torch.tile(
        torch.arange(S).unsqueeze(0),
        (batch_size, S)
    ),
    [batch_size, S, S, 1])
    cell_indices = cell_indices.float()

    norm_x = 1 / S * (best_boxes[..., :1] + cell_indices)
    norm_y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    norm_w_h = 1 / S * best_boxes[..., 2:4]

    norm_boxes = torch.cat([norm_x, norm_y, norm_w_h], dim = -1)
    best_confidences, _ = torch.max(confidence_scores, dim = -1)
    best_confidences = best_confidences.unsqueeze(-1)
    best_classes = torch.argmax(grid_classes, dim = -1).unsqueeze(-1)
    best_classes = best_classes.float()

    norm_predictions = torch.cat([best_classes, best_confidences, norm_boxes], dim = -1)
    return norm_predictions

# Mean Average Precision
