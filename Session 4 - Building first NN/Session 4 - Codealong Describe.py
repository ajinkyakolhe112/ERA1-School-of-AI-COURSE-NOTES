import torch
import torch.nn as nn

import tensorflow.keras as keras
from keras import layers


from torchinfo import summary
import numpy as np


testImage = torch.randn(1,200,200)

print("End")

"""
I have understood the lectures, but not mastered it. 
Go 4 sessions's slides & list essential things. Need to master them. 
Also go through the lecture, and do everything in pytorch. 

people could recall things, but I couldn't. for brain learning, we do need to recall. 
I couldn't identify, things in diagram... It should be instantaneous, not needing to think. 

"""

model.train()
pbar = tqdm(train_loader)
for batch_idx, (data, target) in enumerate(pbar): 
		optimizer.zero_grad()
		output = model(data) 
		loss = F.nll_loss(output, target) 
		loss.backward() 
		optimizer.step() 
		
		pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
	
for batchId, (data,target) in enumerate(dataLoader):
	optimizer.zero_grad()
	yPredicted = model(data)
	lossFun = F.nll_loss(yPredicted,target) # Error Calculate. (NLL_LOSS = funciton of Error)
	loss.backward() # one function call, defined error is backpropogated "L" times through each layer. (graph which is build in forward pass on Data)
	for params in model.get_parameters():
		params = params - params.gradient * learning_rate
		
	output = model(data) # forward operation on data. every operation is building the tree of operations. where derivative is to be calculated
	lossFun = F.nll_loss(output,target)
	lossFun.backward() # Every tensor, which has Gradients set True, has gradient calculation. 
	# gradients are calculated based on the operation, calculated in forward pass. 
	
	# model.weight, model.bias = model.get_named_parameters()
	# model.w = w - w.gradient * learning_rate


for batch_no,(data,target) in enumerate(trainLoader):
	"Current Batch = batch_no"
	
	"z = X*W"
	"activated = sigma(z)"
	data,model,target
	model.state_dict()
	
	optimizer.zero_grad()
	output = model(data) # Builds computational graph & executes forward pass through architecture, with its weights multiplying wiht input. 
	" Error = F(model= X*W,target)"
	lossFn = F.nll_loss(output,target)
	
	lossFn.backward() # graph built in output = model(data), we calculate respective gradient of Error wrt W for given
	model.state_dict()
	
	for param in model.parameters():
		param -= param.grad*0.01
	
	