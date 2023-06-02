from data import getDataLoader
from model import getModel
from utils import trainOneEpoch

import torch

trainDataLoader,testDataLoader = getDataLoader()
model = getModel()

print(model)

device = torch.device("mps")
learning_rate = 0.01
epochs = 2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
ErrorFun = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
	trainOneEpoch(trainDataLoader, epochs , model, ErrorFun, optimizer, device)