import torch
import torch.nn as nn

class customMultiplicationNeuron(nn.Module):
	def __init__():
		super().__init__()
		self.parameter = nn.Parameter()
		
		self.multiplierToLearn = nn.Parameter()
		
		
	def forward(self,inputData):
		outputData = inputData * self.multiplierToLearn
		return outputData

self.parameter.grad
