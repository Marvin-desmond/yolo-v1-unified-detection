import tensorflow as tf 
import numpy as np
import metrics
import pprint 
from loss import YoloLoss
import tqdm 
import metrics

import sys
np.set_printoptions(threshold=sys.maxsize)

import data_loader
import yolo_network

import sys
sys.path.append("../")

import utils # type: ignore
batch_size = 16 

dataset = data_loader.VOCDataLoader()
model = yolo_network.YOLO_Network(use_pretrained = True)
loss = YoloLoss()
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
checkpoint_path = "yolo-v1-{epoch:04d}.ckpt"

epochs = 10
for epoch in range(epochs):
	print(f"\nStart of epoch {epoch}")
	for step, (x_batch_train, y_batch_train) in tqdm.tqdm(
		enumerate(dataset), 
		total = tf.data.experimental.cardinality(dataset).numpy()
		):
		with tf.GradientTape() as tape:
			logits = model(x_batch_train, training=True)
			loss_value = loss(logits, y_batch_train)
		grads = tape.gradient(loss_value, model.trainable_weights, unconnected_gradients=tf.UnconnectedGradients.ZERO)
		optimizer.apply_gradients(zip(grads, model.trainable_weights))
		if step % 200 == 0:
			print(
				"Training loss (for one batch) at step %d: %.4f"
				% (step, float(loss_value))
			)
			print("Seen so far: %s samples" % ((step + 1) * batch_size))
	model.save_weights(checkpoint_path.format(epoch=epoch))
# for images, labels in dataset.take(10):
# 	outputs = model(images)
# 	best_output = metrics.highest_confidence_box(outputs)
# 	best_output = tf.cast(best_output, tf.int64).numpy()
# 	metrics.non_max_suppression(best_output)
# 	loss_value = loss(outputs, labels)
# 	print(f"LOSS VALUE: {loss_value.numpy()}")
# 	print("~"*50)