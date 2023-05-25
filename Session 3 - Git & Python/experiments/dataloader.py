#!/usr/bin/env python3
from torchvision import datasets,transforms

trainDataLoader = torch.utils.data.DataLoader(
	datasets.MNIST("./data",download=True,train=True, transform=transforms.Compose([transforms.ToTensor()])),
	shuffle=True,batch_size=128)

testDataLoader = torch.utils.data.DataLoader(
	datasets.MNIST("./data",download=True,train=False,transform=transforms.Compose([transforms.ToTensor()])),
	shuffle=True,batch_size=128)

for batch_id,(data,target) in enumerate(trainDataLoader):
	print(batch_id,data.shape,target.shape)
	
	break