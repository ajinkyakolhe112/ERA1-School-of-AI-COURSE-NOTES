"MNIST in torchvision"

import torch
from torchvision import datasets, transforms

trainData = datasets.MNIST("data", train=True, download=True)
testData = datasets.MNIST("data", train=False, download=True)

# Transformations are changes you can make to your project data set, after the source data has been processed and loaded
"Existing Data for Vision"
"Increasing DataSet Size / Data Augmentation"
customTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.5, 0.5, 0.5)])

trainData = datasets.MNIST("data", train=True, download=False,transform = customTransforms)

trainDataLoader = torch.utils.data.DataLoader(trainData,batch_size=32,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testData,batch_size=32,shuffle=True)

if __name__=="__main__":
	vars(trainData) # https://share.cleanshot.com/pMd3QvGR
	vars(testData)
	print("Shape of Training Data",trainData.data.shape) # [60000, 28, 28]
	vars(trainDataLoader) # has just dataset property
	
	print("END")
	
	#TODO 1: Figure out how to execute transform function on image datasets
	# trainData = customTransforms(trainData.data) # ERROR
	# IMAGE & TENSOR. (Some functions work on Image, and soem fuctions work on Tensor)
else:
	pass
	
	
	