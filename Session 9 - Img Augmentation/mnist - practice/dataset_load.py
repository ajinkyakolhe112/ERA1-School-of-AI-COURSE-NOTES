import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# TODO: Dataset, Data Augmentation, Viewing Data & Feautures of Data for Classification, Data Loader

train_dataset = datasets.MNIST(download=True,root="./data",train=True, 
	transform = transforms.Compose([
		transforms.ToTensor()
		]))
test_dataset = datasets.MNIST(download=True,root="./data",train=False, 
	transform = transforms.Compose([
		transforms.ToTensor()
		]))

plt.imshow(train_dataset[0][0].squeeze(dim=0))

keyword_value_pair = {
	"batch_size": 64,
	"shuffle": True
}
kwargs = keyword_value_pair

# Data Loader doesn't do transforms
train_loader = torch.utils.data.DataLoader(train_dataset,**kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset,**kwargs)


if __name__ == "__main__":
	len(train_dataset),len(train_dataset)
	total_batches = len(train_dataset) / kwargs["batch_size"]
	for batch_no,(image_tensor,label) in enumerate(tqdm(train_loader)):
		batch_no,image_tensor.shape,label
		pass
	print("END")