from torchvision import datasets,transforms
import torch

"DataSets"
train_dataset = datasets.MNIST('../data', train=True, download=True, 
	transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = datasets.MNIST('../data', train=False, download=True, 
	transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

# Train data transformations
train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1407,), (0.4081,))
    ])

"Data Loaders"
batch_size = 32
kwargs = {'batch_size': batch_size, 'shuffle': True}

train_dataloader = torch.utils.data.DataLoader(train_dataset, **kwargs)
test_dataloader = torch.utils.data.DataLoader(test_dataset, **kwargs)

if __name__ =="__main__":
	# Test
	X,Y = next(iter(train_dataloader))
	print("X.shape,",X.shape, "Y.shape",Y.shape)
