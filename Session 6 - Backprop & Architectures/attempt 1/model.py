import torch
import torch.nn as nn

"Data: [60000, 28, 28]"

class baselineModel(nn.Module):
	def __init__(self):
		super().__init__()
		"4 Blocks of Feature Extraction with increasing features."
		"RF = 9*9. Image = 28*28. Goal RF = 28*28. Delta RF for kernel = 3-1. Goal Depth = 28/2 = 14"
		"RF = Image Size - Output Size (Easier RF Calculation)"
		nn.Conv2d(1,32,(3,3))
		
		nn.Conv2d(32,64,(3,3))

		nn.Conv2d(64,128,(3,3))
		
		nn.Conv2d(32,32,(3,3))
		
		
	def forward(self, dataX):
		pass

model = baselineModel()


if __name__=="__main__":
	model(torch.randn(1,23,23))
	print("END")
	