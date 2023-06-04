import torch
import torch.nn as nn

"Data: [60000, 28, 28]"

class baselineModel(nn.Module):
	def __init__(self):
		super().__init__()
		"4 Blocks of Feature Extraction with increasing features."
		"RF = 9*9. Image = 28*28. Goal RF = 28*28. Delta RF for kernel = 3-1. Goal Depth = 28/2 = 14"
		"RF = Image Size - Output Size (Easier RF Calculation)"
		
		self.conv1 = nn.Conv2d(1,32,(3,3),padding=1)  # RF = 3*3
		self.conv2 = nn.Conv2d(32,32,(3,3),padding=1) # RF = 5*5
		self.pool1 = nn.MaxPool2d(2,2)      # RF = 10*10
		# output = 14*14
		
		self.conv3 = nn.Conv2d(32,64,(3,3),padding=1) # RF = 12*12
		self.conv4 = nn.Conv2d(64,64,(3,3),padding=1) # RF = 14*14
		self.pool2 = nn.MaxPool2d(2,2)	   # RF = 28*28. GOAL RF Reached. 
		# output = 7*7
		
		# Without padding output size = 1*1. Next 2 blocks need 2d data for convolution to extract features.

		self.conv5 = nn.Conv2d(64,128,(3,3))
		# output = 5*5
		
		self.conv6 = nn.Conv2d(128,256,(3,3))
		# output = 3*3, channels = 256
		
		self.fc1 = nn.Linear(256*9,20) # Grouping extracted features into 20 dim vector
		
		self.decision = nn.Linear(20,10) # Decision Maker
		nn.Softmax(dim=1)
		
	"Sequential makes it easier to NNs. To encourage better behaviour, its intentionally made lightweight to encourage overriding forward method"
	def forward(self, dataX):
		outputConv1 = self.conv1(dataX)
		outputConv2 = self.conv2(outputConv1)
		outputPool1 = self.pool1(outputConv2)
		# output = 14*14. Confirmed by executing
		
		outputConv3 = self.conv3(outputPool1)
		outputConv4 = self.conv4(outputConv3)
		outputPool2 = self.pool2(outputConv4)
		# output = 7*7. Confirmed by executing
		
		outputConv5 = self.conv5(outputPool2)
		
		outputConv6 = self.conv6(outputConv5)
		# output = torch.Size([1, 256, 3, 3])
		
		outputReshape = outputConv6.view(-1,256*3*3) # -1 is for Batch
		
		outputFc1 = self.fc1(outputReshape)
		outputDecision = self.decision(outputFc1)
		outputSoftmaxed = torch.nn.functional.softmax(outputDecision,dim=1) # Batch,Columns
		outputSoftmaxed.sum(dim=1) # leaves dim=1
		
		output = torch.nn.functional.log_softmax(outputDecision,dim=1)
		output.sum(dim=1) # this sum isn't 1 unlike above because log. 
		return output

model = baselineModel()


if __name__=="__main__":
	result = model(torch.randn(2,1,28,28))
	print("END")
	