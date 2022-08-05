import pandas as pd
import numpy as np
import tqdm

import sys
sys.path.append("../")

import utils # type: ignore
import transforms

# Importing utils from ../ folder
# ========> from .. import utils ~ used this only for autocompletions
# Import error so import from parent folder using sys.path.append

data = pd.read_parquet("../detections.parquet")

# Removing the 3 classes to remain with only the 20 classes used in the paper
data = data.loc[~data.detection_object.isin(['head', 'hand', 'foot'])]

processing_data = data[['path', 'xmin', 'ymin', 'xmax', 'ymax']]
index_values = processing_data.index.values
processing_data = processing_data.values
data[['x_yolo', 'y_yolo', 'w_norm', 'h_norm']] = 0.0

for index, processing_sample in tqdm.tqdm(
		zip(index_values, processing_data),
		total=data.shape[0]):
	path, *bounding_box = processing_sample
	# A dot is added to go back one directory to find the data
	image_dimensions = (utils.read_image(f'.{path}')).shape[:2]
	norm_bounding_boxes = transforms.b_minmax_to_norm_b_yolo(
		image_dimensions, 
		bounding_box
		)
	norm_bounding_boxes = [round(i, 3) for i in norm_bounding_boxes]
	data.loc[index, ['x_yolo', 'y_yolo', 'w_norm', 'h_norm']
			 ] = norm_bounding_boxes

data.to_parquet("VOC_2012_detections.parquet")