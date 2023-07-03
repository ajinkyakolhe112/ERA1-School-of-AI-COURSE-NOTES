import torch
model = torch.nn.Sequential()

for param in model.parameters():
	"grad is on param level not individual element level. grad is an tensor of same shape as params"
	print(param.shape,param.grad.shape)
	if len(param.shape)>1:
		print("%0.2f"%param.grad)
		for kernelIndex in range(param.shape[0]):
			print(param[kernelIndex])