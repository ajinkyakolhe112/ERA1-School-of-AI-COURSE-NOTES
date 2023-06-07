from torchvision import datasets,transforms
import torch

"DataSets"
train_dataset = datasets.MNIST('../data', 
	train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = datasets.MNIST('../data', 
	train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

"Data Loaders"
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if __name__ =="__main__":
	# Test
	X,Y = next(iter(train_dataloader))
	print("X.shape,",X.shape, "Y.shape",Y.shape)
