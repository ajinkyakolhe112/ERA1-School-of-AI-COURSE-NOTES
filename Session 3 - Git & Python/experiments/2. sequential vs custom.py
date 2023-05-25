#!/usr/bin/env python3

# You can't do any branching at all. And branching is very powerful. So we use alternative.

"""
Model Architecture & Parameters
Data fed Batch by Batch

1. initialization of layers & parameters
1. FOR Data fed batch by batch
2. FOR forward pass layer by layer
3. error calculation at last layer. (Activation Function Choosen)
4. FOR Error back propogation reverse layer by layer
5. Weight Update accordingly
"""

"""
A Single Neuron we put. It's a building block of learning. Each Single Neuron is just X_batch \odot W_neuronNo
Every Single Neuron, according to Error Value, corrects its individual weights, according to each's contribution to Error
W = horizontal array of weights. 


Human Brain Hz = one Cycle in seconds. One second calculations
Computer Brain Hz = one Cycle in second. One second calculations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F # functional module is all the Classes in nn module but in format of functions. Simple functions. F.relu F.selu() F.conv2d
import torch.nn.functional as functions

## Layers in pytorch, actually are nothing but functions from functional module being called with inputX * Weight + Bias. 
## Layer is a class instantiated with its internal variables or parameters being updated for all batches of data. 


class FirstDNN(nn.Module):
	def __init__(self): # Initialization for layers to be used
		super().__init__()
		nn.Conv2d() # Processing & Parameters... Parameters initialize. 
		nn.ReLU() # Don't need to create. Non parameter layer. Just processing layer. 
		nn.Conv2d()
		CustomReshapeImageto1D() # For Vision. Somewhere from Image * Channel view, we will go into vector space of integers for further processing Generally final block of 4 block structure NN
		
	# extending method 
	def forward(self,inputData)# single Example.. one by one x_1, x_2 or array of batch of datapoints # this will have hooks
		firstTransformedData = Conv2d(inputData)
		secondTransformedData = Conv2d(firstTransformedData)
		
	def backward(ErrorFunction): # will probably have hooks
		According to dynamic operation in above forward, we have library which can calculate this. why do we have that library, because we have fast computers which can calculate this. 
		
class CustomReshape(nn.Module):
	def __init__(self,newShape):
		self.newShape = newShape
		
	def forward(self,inputData):
		torch.nn.functional.view
		