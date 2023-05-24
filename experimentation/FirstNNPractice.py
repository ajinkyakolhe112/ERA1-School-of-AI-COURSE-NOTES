#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

"DataSet, DataLoader, Model & Model Parameters, Training the Model & Evaluation via Accuracy, Improvement of Model"
from torchvision import datasets 
from torchvision import transforms
from torchvision import datasets as visionDatasets 
from torchvision import transforms as visionTransforms

transformList = visionTransforms.Compose([visionTransforms.ToTensor(),])

MnistTrainingDataset = visionDatasets.MNIST("./data",download=True,train=True, transform = transformList, target_transform = None)
MnistTestingDataset = visionDatasets.MNIST("./data",download=True,train=False, transform = transformList, target_transform = None)

TrainLoader = torch.utils.data.DataLoader(MnistTrainingDataset, shuffle=True,batch_size = 128)
TestLoader = torch.utils.data.DataLoader(MnistTestingDataset, shuffle=True, batch_size = 128)

for batchId , (data,target) in enumerate(TrainLoader): # data = (BatchSize, Channels, Width, Height) , target = (BatchSize)
	print(batchId,data.shape,target.shape)
	"X & y","prediction = f(x,W)"
	break

# Input Image = 1, 28, 28
class customNetwork(nn.Module):
	def __init__(self,): # "initialization of transforming layers & parameters"
		super().__init__()
		self.conv1 = nn.Conv2d(1,128,(3,3)) # initializes weights & bias for a conv2d layer
		self.conv2 = nn.Conv2d(128,252,(3,3))		
		self.pool = nn.MaxPool2d((2,2))
		self.conv31 = nn.Conv2d(252,252,(3,3))
		self.conv32 = nn.Conv2d(252,32,(3,3))
		self.pool = nn.MaxPool2d((2,2))
		self.conv4 = nn.Conv2d(32,10,(1,1)) 		# torch.Size([10, 3, 3])
		
		self.fc1 = nn.Linear(10*3*3,10)
		
		#print(self.conv1.weight.shape,self.conv1.bias.shape)
		#print(self.conv2.weight.shape,self.conv2.bias.shape)
		
	
	# Iteratively build layer by layer transformation & accordingly initialize its parameters
	def forward(self,inputData: torch.Tensor): # data being processed through transforming layers
		"data to feature map extraction","Human vision processing happens in 4 blocks based on features of image"
		transform = F.relu(self.conv1(inputData))
		transform = F.relu(self.pool(self.conv2(transform)))
		transform = F.relu(self.conv32(self.pool(self.conv31(transform))))
		transform = F.relu(self.conv4(transform))
		# torch.Size([10, 3, 3])
		

		transform = torch.flatten(transform) # torch.flatten or torch.reshape or tensor.view method # Converts to 1d
		#transform = transform.view(1,-1)
		transform = self.fc1(transform)
		
		
		output = F.log_softmax(transform)
		
		print(transform.shape)
		
		"feature map to decision making"
		
		pass

model = customNetwork()
model(torch.randn(1,28,28))

print("End")