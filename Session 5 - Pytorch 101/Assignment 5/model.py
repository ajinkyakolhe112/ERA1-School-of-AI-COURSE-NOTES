import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size=(3,3))
		self.conv2 = nn.Conv2d(32,64,3)
		self.conv3 = nn.Conv2d(64,128,3)
		self.conv4 = nn.Conv2d(128,256,3)
		# Output: torch.Size([1, 256, 4, 4])"
		
		self.fc1 = nn.Linear(256*4*4,50)
		self.fc2 = nn.Linear(50,10)
	
	def forward(self,DataX):
		transformed = self.conv1(DataX)
		transformed = F.relu(transformed)
		
		transformed = self.conv2(transformed)
		transformed = F.max_pool2d(transformed,2)
		transformed = F.relu(transformed)
		
		x = F.relu(self.conv3(transformed), 2)
		x = F.relu(F.max_pool2d(self.conv4(x), 2))
		# Output: torch.Size([1, 256, 4, 4])"
		
		x = x.view(-1,256*4*4)
		x = self.fc1(x)
		x = F.relu(x)
		
		output = self.fc2(x)
		outputProb = F.log_softmax(output,dim=1)
		
		return outputProb
		
		print("END")
		pass

testModel = Net()
testModel(torch.randn(1,1,28,28))