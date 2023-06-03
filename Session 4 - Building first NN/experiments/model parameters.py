import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchinfo import summary

summary(nn.Sequential(), input_size=(1, 28, 28), verbose=2,
		col_names=["input_size", "output_size", "kernel_size", "mult_adds", "num_params", "params_percent"], col_width=20);


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device


class Net(nn.Module):
	#This defines the structure of the NN.
	def __init__(self):
		super(Net, self).__init__()
		# 4 blocks of 4 conv layers each extracting increasing in number & higher depth features each
		# 2 max pooling & 4 conv layers. (image size after: (28 - 4*2)/2*2
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 26*26
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 24*24, 12*12
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3)  # 10*10
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3)  # 8*8, 4*4
		# output Image = (256,5,5)
		# check transform stage in forward
		# self.fc0 = nn.Linear(256*4*4, 256*4*4)
		self.fc1 = nn.Linear(256 * 4 * 4,
							50)  # singleImage = 256 Channels. Size = 4*4. Converted to Vector. Output = (1*50) or (1,50)
		self.fc2 = nn.Linear(50, 10)  # Output = (1*10) or (1,10)
		self.fc0 = nn.Sequential() #nn.Linear(256 * 4 * 4, 256 * 4 * 4)
		
	def forward(self, x):
		x = F.relu(self.conv1(x), 2)  # 2 here is inplace=False.
		x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 2 = Kernel Size here.
		x = F.relu(self.conv3(x), 2)
		x = F.relu(F.max_pool2d(self.conv4(x), 2))
		
		# output Image = (-1,256,5,5)
		# incoming vector = (-1,1,Multiplication)
		x = x.view(-1, 1, 256 * 4 * 4)
		x = F.relu(self.fc1(self.fc0(x)))  # RELU? Extracted features are being discarded may be?
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)  # F.softmax(x)[0].sum()
		return x
	
	
model = Net()


summary(model, input_size=(1, 28, 28), verbose=2,
		col_names=["input_size", "output_size", "kernel_size", "mult_adds", "num_params", "params_percent"], col_width=20);
