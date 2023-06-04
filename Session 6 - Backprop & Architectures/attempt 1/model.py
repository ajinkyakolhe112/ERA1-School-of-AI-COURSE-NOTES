import torch
import torch.nn as nn

class baselineModel(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, dataX):
		pass

model = baselineModel()

if __name__=="__main__":
	model(torch.randn(1,23,23))
	print("END")
	