from datasetDataloader import getData
from modelArchitecture import getModel,customModel
"E = YPrecited - Y = f(X,W)"
"Adusting W according to E"
"Monitoring E"

import torch

def trainModel(DataUtil, ModelUtil):
	model = ModelUtil.model
	train_dataloader = DataUtil.trainLoader
	test_dataloader = DataUtil.testLoader
	
	historyLoss = []
	for batch_id,(dataX,YActual) in enumerate(train_dataloader):
		dataX, YActual = dataX.to(device), YActual.to(device)
		optimizer.zero_grad()
		
		yPredicted = model(dataX) # computational graph is built here. With every operation being recorded, it turns out to be a tree like structure. 
		try:
			print(model.state_dict())
			for name,param in model.named_parameters():
				print("name",param.shape,param.grad)
		except:
			pass
		lossFunction = torch.nn.CrossEntropyLoss()
		lossSingleValue = lossFunction(yPredicted,YActual) # yPredicted = f(X,W)
		lossSingleValue.backward()
		try:
			for name,param in model.named_parameters():
				print("name",param.shape,param.grad)
		except:
			pass
			
		for param in model.parameters():
			"grad is on param level not individual element level. grad is an tensor of same shape as params "
			print(param.shape,param.grad.shape)
			if len(param.shape)>1:
				print("%0.2f"%param.grad)
				for kernelIndex in range(param.shape[0]):
					print(param[kernelIndex])
					
		optimizer.step()
		
		historyLoss.append(lossValue.item())
		"""
		for i,param in enumerate(model.parameters()):
			print("In Layer %f",i/2)
			print(param.shape)
			param = param - param.grad*learning_rate
		"""
		
		if batch_id==10:
			break
	
from dataclasses import dataclass
@dataclass
class modelUtils:
	model: torch.nn.Module
	optimizer: torch.optim.Optimizer
	learning_rate: float

model = customModel()
device = torch.device("mps")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
learning_rate = 0.01
modelUtil = modelUtils(model,optimizer,learning_rate)
dataUtil = getData()

trainModel(dataUtil,modelUtil)

print("END")