from matplotlib.pyplot import box
import tensorflow as tf 
import metrics
import numpy as np 

class YoloLoss(tf.keras.layers.Layer):
	"""
	Calculate the loss for yolo (v1) model
	"""
	def __init__(self, S=7, B=2, C=20):
		super(YoloLoss, self).__init__()
		self.mse = tf.keras.losses.MeanSquaredError(reduction="sum") 
		self.S = S
		self.B = B
		self.C = C
		self.lambda_coord = 5
		self.lambda_noobj = 0.5

	def call(self, predictions, target):
		if_one_cells = predictions[..., 21:25]   
		if_two_cells = predictions[..., 26:30]   
		target_cells = target[..., 21:25]
		# Calculate IoU for the two predicted bounding boxes with target bbox
		iou_one = metrics.intersection_over_union(if_one_cells, target_cells)
		iou_two = metrics.intersection_over_union(if_two_cells, target_cells)
		iou_both = tf.concat([iou_one, iou_two], axis = -1)

		I_obj_i = target[..., 20:21]
		I_obj_ij = tf.expand_dims(tf.math.argmax(iou_both, axis = -1), axis = -1)
		I_obj_ij = tf.cast(I_obj_ij, tf.float32)
		# if best_index is 0, then 1 - 0 will choose the first, then delete the second (0 * second)
		prediction_cells = (1 - I_obj_ij) * if_one_cells + I_obj_ij * if_two_cells

		x_y = target_cells[..., 0:2]
		w_h = target_cells[..., 2:4]
		x_y_hat = prediction_cells[..., 0:2]
		w_h_hat = prediction_cells[..., 2:4]
		x_y = tf.reshape(x_y, [-1, x_y.shape[-1]])
		w_h = tf.reshape(w_h, [-1, w_h.shape[-1]])
		x_y_hat = tf.reshape(x_y_hat, [-1, x_y_hat.shape[-1]])
		w_h_hat = tf.reshape(w_h_hat, [-1, w_h_hat.shape[-1]])

		# ======================== #
		#         BOX LOSS         #
		# ======================== #
		# regression loss between the grid cell offsets
		loss_x_y = self.mse(x_y, x_y_hat)
		# regression loss between the width and height values
		loss_w_h = self.mse(
			tf.sqrt(w_h), 
			tf.sign(w_h_hat) * tf.sqrt(tf.abs(w_h_hat + tf.constant(1e-6)))
			)
		box_loss = loss_x_y + loss_w_h
		# ==================== #
		#      OBJECT LOSS     #
		# ==================== #
		c_i = target[..., 20:21]
		c_i_hat = (1 - I_obj_ij) * predictions[..., 20:21] + I_obj_ij * predictions[..., 25:26]
		c_i = tf.reshape(c_i, [-1])
		c_i_hat = tf.reshape(c_i_hat, [-1])
		object_loss = self.mse(c_i, c_i_hat)
		# ==================== #
		#    NO OBJECT LOSS    #
		# ==================== #
		# I_noobj_ij = 1 - I_obj_ij
		c_i_hat_ = I_obj_ij * predictions[..., 20:21] + (1 - I_obj_ij) * predictions[..., 25:26]
		c_i_hat_ = tf.reshape(c_i_hat_, [-1])
		no_object_loss = self.mse(c_i, c_i_hat_)
		# ================== #
		#     CLASS LOSS     #
		# ================== #
		p_c_predictions = I_obj_i * predictions[..., :20]
		p_c_target = I_obj_i * target[..., :20]
		class_loss = self.mse(p_c_target, p_c_predictions)
		
		loss = (
			self.lambda_coord * box_loss
			+ object_loss
			+ self.lambda_noobj * no_object_loss
			+ class_loss
		)
		return loss



