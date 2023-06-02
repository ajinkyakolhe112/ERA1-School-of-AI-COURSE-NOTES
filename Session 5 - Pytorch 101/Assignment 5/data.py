from torchvision import datasets, transforms
import torch

"DataSet = MNIST -> DataSet Transforms -> DataLoader"

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
	transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
	transforms.Resize((28, 28)),
	transforms.RandomRotation((-15., 15.), fill=0),
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
	])

train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)


batch_size = 512
kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

test_loader = torch.utils.data.DataLoader(train_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
