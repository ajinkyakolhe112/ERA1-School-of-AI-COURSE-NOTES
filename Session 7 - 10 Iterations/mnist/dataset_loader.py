import torch
from torchvision import datasets, transforms

# vars(transform_pipeline).keys() = ['transforms']
# dir(transform_pipeline) = ['transforms']
transform_pipeline = transforms.Compose(
	[
		transforms.ToTensor(),
	]
)

# vars(train_data).keys() =	['root', 'transform', 'target_transform', 'transforms', 'train', 'data', 'targets'])
# dir(train_data) =			['root', 'data', 'targets', 'classes', 'transform' , 'target-transform', 'train' ] | ['transforms']
train_data = 	datasets.MNIST(root = "./data", download=True, train=True , transform=transform_pipeline)
test_data = 	datasets.MNIST(root = "./data", download=True, train=False, transform=transform_pipeline)

key_value_pair_dict = {"shuffle":True, "batch_size":128 }
kwargs = key_value_pair_dict

train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
test_loader  = torch.utils.data.DataLoader(test_data , **kwargs)

if __name__ == "__main__":
	for batch_no,(data,target) in enumerate(train_loader):
		data.shape,target.shape
		break
	print("END")