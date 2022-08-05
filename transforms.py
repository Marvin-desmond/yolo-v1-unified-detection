import pandas as pd 
import numpy as np 
 
from typing import Optional, List as list, Tuple as tuple
def b_minmax_to_b_center(
	xymin_xymax: tuple[int, int, int, int]
	) -> tuple[int, int, int, int]:
	"""
	This functions converts bounding boxes from
	x_min, y_min, x_max, y_max (un-normalized) 
	to
	x_center, y_center, width, height (un-normalized)
	format
	"""
	xmin, ymin, xmax, ymax = xymin_xymax
	width = xmax - xmin
	height = ymax - ymin
	x_center = int(xmin + width / 2.0)
	y_center = int(ymin + height / 2.0)
	return (x_center, y_center, width, height)

def inverse_b_minmax_to_b_center(
	xy_center_wh: tuple[int, int, int, int],
)-> tuple[int, int, int, int]:
	"""
	This functions converts bounding boxes from
	x_center, y_center, width, height (un-normalized)
	to
	x_min, y_min, x_max, y_max (un-normalized) 
	format
	"""
	x_center, y_center, width, height = xy_center_wh
	xmin = int(x_center - width / 2.0)
	ymin = int(y_center - height / 2.0)
	xmax = width + xmin
	ymax = height + ymin 
	return (xmin, ymin, xmax, ymax)


def b_center_to_b_normalized(
	image_dimensions: tuple[int, int],
	xy_center_wh: tuple[int, int, int, int]
) -> tuple[float, float, float, float]:
	"""
	This function transforms bounding boxes
	x_center, y_center, width, height (un-normalized)
	format to
	x_center, y_center, width, height (normalized)
	w.r.t. the image height and width dimensions.
	"""
	image_height, image_width = image_dimensions
	x_center, y_center, width, height = xy_center_wh
	norm_x = x_center / image_width
	norm_y = y_center / image_height
	norm_width = width / image_width
	norm_height = height / image_height
	return (norm_x, norm_y, norm_width, norm_height)

def inverse_b_center_to_b_normalized(
	norm_xy_center_wh: tuple[float, float, float, float],
	image_dimensions: Optional[tuple[int, int]] = (448, 448),
	) -> tuple[int, int, int, int]:
	"""
	This function transforms bounding boxes
	x_center, y_center, width, height (normalized)
	w.r.t. the image height and width dimensions.
	format to
	x_center, y_center, width, height (un-normalized)
	"""
	norm_x, norm_y, norm_width, norm_height = norm_xy_center_wh
	image_height, image_width = image_dimensions
	x_center = int(norm_x * image_width)
	y_center = int(norm_y * image_height)
	width = int(norm_width * image_width)
	height = int(norm_height * image_height)
	return (x_center, y_center, width, height)
	
def b_normalized_to_b_yolo(
	norm_xy_center_wh: tuple[int, int, int ,int],
	S: int = 7
	) -> tuple[float, float, float, float]:
		"""
		This function transforms normalized bounding boxes 
		from w.r.t image dimensions to normalized dimensions 
		w.r.t each grid cell bounding it.
		"""
		norm_x, norm_y, norm_width, norm_height = norm_xy_center_wh
		# Calculation of lower bounds of the grid cell
		grid_x = np.floor(S * norm_x)
		grid_y = np.floor(S * norm_y)
		# Calculation of the offset of XY centers
		# From the lower bounds of the grid cell
		yolo_x: float = S * norm_x  - grid_x
		yolo_y: float = S * norm_y - grid_y
		return (yolo_x, yolo_y, norm_width, norm_height)


def b_minmax_to_norm_b_yolo(
	image_dimensions: tuple[int, int],
	b_minmax: tuple[int, int, int, int],
	S: int = 7) -> tuple[float, float ,float, float]:
	"""
	This function converts bounding boxes 
	from 
	x_min, y_min, x_max, y_max (un-normalized)
	to
	x_center, y_center, width, height (normalized w.r.t grid cell)
	format
	through
	x_center, y_center, width, height (normalized w.r.t image) 
	"""
	# from b_minmax to norm_b_center
	inverse_image_height = 1 / image_dimensions[0]
	inverse_image_width = 1 / image_dimensions[1]
	x_min, y_min, x_max, y_max = b_minmax 
	norm_x =  (x_min + x_max) / 2.0 * inverse_image_width
	norm_y = (y_min + y_max) / 2.0 * inverse_image_height 
	norm_width = (x_max - x_min) * inverse_image_width
	norm_height = (y_max - y_min) * inverse_image_height
	# now from norm b_center to norm_b_yolo
	grid_x = np.floor(S * norm_x)
	grid_y = np.floor(S * norm_y)
	yolo_x: float = S * norm_x - grid_x
	yolo_y: float = S * norm_y - grid_y 
	return (yolo_x, yolo_y, norm_width, norm_height)

def norm_b_yolo_to_b_minmax(
	image_dimensions: tuple[int, int],
	norm_b_yolo: tuple[float, float ,float, float],
	S: int = 7
) -> tuple[int, int, int, int]:
	pass