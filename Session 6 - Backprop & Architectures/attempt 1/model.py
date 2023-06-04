import torch
import torch.nn as nn

"Data: [60000, 28, 28]"

class baselineModel(nn.Module):
	def __init__(self):
		super().__init__()
		"4 Blocks of Feature Extraction with increasing features."
		"RF = 9*9. Image = 28*28. Goal RF = 28*28. Delta RF for kernel = 3-1. Goal Depth = 28/2 = 14"
		"RF = Image Size - Output Size (Easier RF Calculation)"
		nn.Conv2d(1,32,(3,3),padding=1)  # RF = 3*3
		nn.Conv2d(32,32,(3,3),padding=1) # RF = 5*5
		nn.MaxPool2d(2,2)      # RF = 10*10
		# output = 14*14
		
		nn.Conv2d(32,64,(3,3),padding=1) # RF = 12*12
		nn.Conv2d(64,64,(3,3),padding=1) # RF = 14*14
		nn.MaxPool2d(2,2)	   # RF = 28*28. GOAL RF Reached. 
		# output = 7*7
		
		# Without padding output size = 1*1. Next 2 blocks need 2d data for convolution to extract features.

		nn.Conv2d(64,128,(3,3))
		# output = 5*5
		
		nn.Conv2d(128,256,(3,3))
		# output = 3*3
		
		nn.Linear(9,20) # Grouping extracted features into 20 dim vector
		
		nn.Linear(20,10) # Decision Maker
		nn.Softmax(dim=1)
		
	def forward(self, dataX):
		
		pass

model = baselineModel()


if __name__=="__main__":
	model(torch.randn(1,23,23))
	print("END")
	