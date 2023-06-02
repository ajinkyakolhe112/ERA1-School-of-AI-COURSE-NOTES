from data import getDataLoader
from model import getModel
from model import customModel
from prettytable import PrettyTable

"E = YPrecited - Y = f(X,W)"
"Adusting W according to E"
"Monitoring E"

import torch

def investigate_Gradients_Parameters(model,layer):
	Weights = model.state_dict()
	print(model.state_dict())
	for name,param in model.named_parameters():
		print("name",param,param.shape)
		parameterValues = param
		gradientsError = param.grad # "Parameters after Update Acc to Error Contribution"
		
		
def trainOneBatch(BatchId, DataX,YActual,model, ErrorFun, optimizer, device):
	print("Executing Batch",BatchId)
	
	DataX, YActual = DataX.to(device), YActual.to(device)
	
	YPredicted = model(DataX) # YPredicted = f(X,W). Computational graph is built here. With every operation being recorded
	ErrorValue = ErrorFun(YActual,YPredicted)
	ErrorValue.backward()
	investigate_Gradients_Parameters(model)
	"""
	for i,param in enumerate(model.parameters()):
		print("In Layer %f",i/2)
		print(param.shape)
		param = param - param.grad*learning_rate
	"""
	optimizer.step()
	optimizer.zero_grad()
	
	return ErrorValue

"train_dataloader:torch.utils.DataLoader, model:torch.nn.Module, ErrorFun:torch.nn.Module, optimizer: torch.optim.Optmizer, device:torch.device)"
def trainOneEpoch(train_dataloader, epochs , model, ErrorFun, optimizer, device):
	historyLoss = []
	print("Executing Epoch No",epochs)
	model.to(device)
	
	for batch_id,(dataX,YActual) in enumerate(train_dataloader):
		dataX, YActual = dataX.to(device), YActual.to(device)
		
		lossSingleValue = trainOneBatch(batch_id, dataX,YActual,model, ErrorFun, optimizer, device)
		
		historyLoss.append(lossValue.item())
		if batch_id==10:
			break
	
def count_parameters(model):
	table = PrettyTable(["Modules", "Parameters"])
	total_params = 0
	
	for name, parameter in model.named_parameters():
		if parameter.requires_grad == True:
			paramNum = parameter.numel()
			table.add_row([name, paramNum])
			total_params+=paramNum
			
	print(table)
	print(f"Total Trainable Params: {total_params}")
	return total_params

