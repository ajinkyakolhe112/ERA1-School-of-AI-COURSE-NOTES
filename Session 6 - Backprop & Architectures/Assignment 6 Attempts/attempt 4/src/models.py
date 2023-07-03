import torch
import torch.nn as nn
from torch.nn.functional import relu

class FirstDNN(nn.Module):
	def __init__(self):
		super(FirstDNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
		self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.conv5 = nn.Conv2d(256, 512, 3)
		self.conv6 = nn.Conv2d(512, 1024, 3)
		self.conv7 = nn.Conv2d(1024, 10, 3)
	
	def forward(self, x):
		x = self.pool1(relu(self.conv2(relu(self.conv1(x)))))
		x = self.pool2(relu(self.conv4(relu(self.conv3(x)))))
		x = relu(self.conv6(relu(self.conv5(x))))
		x = relu(self.conv7(x))
		x = x.view(-1, 10)
		x = nn.functional.log_softmax(x,dim=1)
		return x

if __name__ =="__main__":
	# Test
	img = torch.randn(7,1,28,28)
	
	model = FirstDNN()
	output = model(img) # Forward Pass
	print("Forward Pass output",output.shape)
