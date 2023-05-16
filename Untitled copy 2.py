#!/usr/bin/env python3

import torch
import torch.nn as nn

from torchinfo import summary
from torchvision import datasets,transforms
from sklearn import datasets as toysets

testScalar = torch.tensor(10) # a cell
testVector = torch.randn(10) # (10,1)
testImage = torch.randn(10,10)
testImageTensor = torch.randn(10,28,28) # 2 dimentional h*w image in c channels. Channel is 3rd dimention

class customNN(nn.Module):
	def __init__():
		super().__init__()
		self.conv1 = nn.Conv2d(input_Channels,output_Channels,(3,3))
	
		
	def forward(self,inputData):
		firstLayerOutput = self.conv1(inputData) # CHW
		pass


keras.Sequential(
	layers.Dense(units=1,activation="relu"),
	
)
	

trainDataLoader = torch.utils.data.DataLoader(
	datasets.MNIST("./data",download=True,train=True, transform=transforms.Compose([transforms.ToTensor()])),
	shuffle=True,batch_size=128)

testDataLoader = torch.utils.data.DataLoader(
	datasets.MNIST("./data",download=True,train=False,transform=transforms.Compose([transforms.ToTensor()])),
	shuffle=True,batch_size=128)




for batch_id,(data,target) in enumerate(trainDataLoader):
	print(batch_id,data.shape,target.shape)
	
	break


	

print("End")