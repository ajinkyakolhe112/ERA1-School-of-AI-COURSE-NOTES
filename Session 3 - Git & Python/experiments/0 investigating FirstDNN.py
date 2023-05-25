#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

model = nn.Sequential(
	nn.Conv2d(1,128,3),
	nn.ReLU(),
	nn.Conv2d(128,128,3),
	nn.ReLU(),
	
)
singleImage = torch.randn(1,28,28)

flattenLayer = torch.nn.Flatten()
vars(flattenLayer) # Flatten Layer doesn't have weights... Not every layer is transformation. Max Pooling, Flattening, Conv1D are different kind of layers. 

# Making it work All Again
model = nn.Sequential(
	nn.Conv2d(1,128,3),
	nn.ReLU(),
	nn.Conv2d(128,128,3), # Single Example 28*28 with certain channels
	nn.ReLU(),
	nn.Flatten() # Single Example 28*28 flatten. 128 Channels with 24*24...  
	
)
output = model(singleImage)
output, output.shape
vars(output)
dir(output)

#%%
example = torch.randn(1,10)
model = nn.Sequential(
	nn.Softmax()
)
output = model(singleImage)
output, output.shape