import torch
import torchvision

def get_data():
	Train_Dataset = torchvision.datasets.MNIST(root = './data',train = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1307,), std = (0.3081,))]), download = True)
	Test_Dataset = torchvision.datasets.MNIST(root = './data',train = False,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),download=True)

	Train_Loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = True)
	Test_Loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = BATCH_SIZE,shuffle = True)

	return Train_Dataset,Test_Dataset,Train_Loader,Test_Loader

# Naming: Train_DataLoader is too verbose