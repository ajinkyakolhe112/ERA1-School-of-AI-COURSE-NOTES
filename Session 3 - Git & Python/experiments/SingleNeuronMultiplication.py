import torch
import torch.nn as nn

class customMultiplicationNeuron(nn.Module):
	def __init__():
		super().__init__()
		self.parameter = nn.Parameter()
		
		self.multiplier_to_learn = nn.Parameter()
		
		
	def forward(self,input_data):
		output_data = input_data * self.multiplier_to_learn
		return output_data

self.parameter.grad
