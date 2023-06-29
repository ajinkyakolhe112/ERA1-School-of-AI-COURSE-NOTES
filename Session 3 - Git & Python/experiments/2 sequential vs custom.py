#!/usr/bin/env python3

# You can't do any branching at all. And branching is very powerful. So we use alternative.

#%% [markdown]
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
#%% [markdown]
"""
- A Single Neuron we put. It's a building block of learning. Each Single Neuron is just $X\_batch \odot W\_neuron\_no$
- Every Single Neuron, according to Error Value, corrects its individual weights, according to each's contribution to Error
- W = horizontal array of weights. 
- Human Brain Hz = one Cycle in seconds. One second calculations
- Computer Brain Hz = one Cycle in second. One second calculations
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F # functional module is all the Classes in nn module but in format of functions. Simple functions. F.relu F.selu() F.conv2d
import torch.nn.functional as functions

## Layers in pytorch, actually are nothing but functions from functional module being called with inputX * Weight + Bias. 
## Layer is a class instantiated with its internal variables or parameters being updated for all batches of data. 
