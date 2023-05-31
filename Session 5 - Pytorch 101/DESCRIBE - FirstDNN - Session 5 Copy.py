#!/usr/bin/env python3
"Describe forces recall. Helps in learning"

"LOW LEVEL DL CODE vs DL as API"
"for on frontier of experimentation, you need a low level library"
"for DL as API, you will not be able to innovate much" + "for serious competetior advantage, need custom control"


"DL = Data + Model. Learned Model is valuable. Very very valuable."
"Data, Model => Dataset, DataLoader, Model Architecture, Model Learning. => Loss Function, "
# QUESTION: Ask, where do you think you can improve the NN performance?

import tensorflow as tf
import numpy as np
dir(torch),dir(tf), dir(np), dir(tf.keras)

import torch # dir() here will tell torch is imported. Memory can be calculated by 
import torch.nn as nn # either import as nn or use torch.nn directly. Eitherway nn is torch.nn
#IMP: how torch.nn works really tutorial. Writing same NN in 7 different ways in increamental use of .nn module in pytorch
import torch.nn.functional as F # all functional equivalent of torch.nn. A place were all functions in nn umbrela live
import torch.optim as optim # A general optimizer. Precise control over how to optmize something
from torchvision import datasets, transforms # torchvision.datasets all group
#TODO: (MNIST, CIFAR, IMAGENET. LeNet, Alexnet, VggNet, Inception) 3 Datasets, 4 Architectures
#!pip install torchsummary
from torchsummary import summary # superceed by torchinfo. All PARAMETERS calculation library. 
#TODO: Implement this from scratch, for precise printing of model parameters according to customization. Will also help in understanding parametes better.
# torchinfo summary documentation. discover,read & understand.(input_size,output_size,kernel_size). Print these operations in latex Maths

use_cuda = torch.cuda.is_available() # nvidia has optimized & accelerated matrix multiplication. Threads & Cores
#TODO: AI Accelerators. Hardware, Memory Wall. ARM Chip M1. 
device = torch.device("cuda" if use_cuda else "cpu")
device # device. Important feature in Tensor Variable. Parameters & Weights need to be on same device.
# cpu, cuda or tpu or mps. MPS = Metal Performance Shaders

"torch cuda device vs torch.backends.cudnn. EXCELLENT ANSWER BY BARD"

#%%
# torch.datasets has dataset utility for vision. Datasets generally need to be augmented, because more data always helps. Because learning efficiency is quite low. Delta of Error Per Batch would be learning Efficiency. 
# torch.dataloader. A minimal batch data loader. simple function. 

# QUESTION: Should transform belong to datasets or dataloader??? Makes you consider placement & function of both datasets & dataloader


batch_size = 1024

# SIMD: SINGLE INSTRUCTION on MULTIPLE DATA. Accelerate by number of cores?
# pytorch + cuDNN premitives used in Pytorch + cuda for custom programming
# Threads would be on each core, that would the size
# batch_size: This would be parallelization.
# Actual Operation is happening here. 
#TODO: Study GPU & parallelization & custom Accelerators

train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=True, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(), # Standardize. Could be Named ToTensor() Range: [0-1)
						transforms.Normalize((0.1307,), (0.3081,)) # Is Order important? I think Normalize & ToTensor could be anywhere and still it would be fine.
					])),
	batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('../data', train=False, transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					])),
	batch_size=batch_size, shuffle=True)

#%%
from tqdm import tqdm
pbar = tqdm(train_loader)

for batch_idx, (data, target) in enumerate(pbar): # enumerate(tqdm(train_loader)) 
	data, target = data.to(device), target.to(device)
	print("One Batch Ends")

#%%
class FirstDNN(nn.Module): # 1. state_dict(), model.parameters(), model.named_parameters()
	def __init__(self):
		super(FirstDNN, self).__init__()
		# r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out:28, j_out:1
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.pool1 = nn.MaxPool2d(2, 2)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.pool2 = nn.MaxPool2d(2, 2)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv5 = nn.Conv2d(256, 512, 3)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv6 = nn.Conv2d(512, 1024, 3)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv7 = nn.Conv2d(1024, 10, 3)
# Correct values
# https://user-images.githubusercontent.com/498461/238034116-7db4cec0-7738-42df-8b67-afa971428d39.png
	def forward(self, x):
		x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
		x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
		x = F.relu(self.conv6(F.relu(self.conv5(x))))
		x = F.relu(self.conv7(x))
		x = x.view(-1, 10)
		return F.log_softmax(x)

#%%
model = FirstDNN().to(device)
summary(model, input_size=(1, 28, 28))

#%%
from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
	model.train()
	pbar = tqdm(train_loader)
	for batch_idx, (data, target) in enumerate(pbar):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		pbar.set_description('loss=%0.2f batch_id=%i'%loss.item(),batch_idx )
		
		
def test(model, device, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			
	test_loss /= len(test_loader.dataset)
	
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))
	
#%%
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 2):
	train(model, device, train_loader, optimizer, epoch)
	test(model, device, test_loader)