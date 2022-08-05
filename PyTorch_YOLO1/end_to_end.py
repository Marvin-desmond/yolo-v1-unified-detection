import torch
from torch.utils.data import DataLoader
import numpy as np
import metrics
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

dataset = data_loader.VOCDatasetLoader()
dataset_loader = DataLoader(dataset, batch_size=16, shuffle=True)
model = yolo_network.YOLO_Network(use_pretrained = True)
criterion = YoloLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-7)

model.train()
epochs = 2
for epoch in range(epochs):
	running_loss = 0.0
	for i, data in tqdm.tqdm(enumerate(dataset_loader, 0), total=len(dataset_loader)):
		images, labels = data
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i % 200 == 0:    # print every 200 mini-batches
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
			running_loss = 0.0

print('Finished Training')
