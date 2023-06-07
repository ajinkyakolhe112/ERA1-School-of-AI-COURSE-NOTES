#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#!pip install torchsummary
from torchsummary import summary
from tqdm import tqdm

use_cudaFlag = torch.cuda.is_available()
device = torch.device("cuda" if use_cudaFlag else "mps")
device

#%%
"DataSets"
train_dataset = datasets.MNIST('../data', 
	train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
test_dataset = datasets.MNIST('../data', 
	train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))

"Data Loaders"
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(train_dataset	, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

for batch_idx, (data, target) in enumerate(train_dataLoader):
	data, target
	print("One Batch Ends")
	break

"EXPORTS: train_dataLoader,test_dataLoader with batch_size"

#%%
class FirstDNN(nn.Module):
	def __init__(self):
		super(FirstDNN, self).__init__()
		# r_in:1, n_in:28, j_in:1, s:1, r_out:3, n_out:28, j_out:1
		"EACH LAYER: (ENTIRE INCOMING DATA For EACH NEURON, NUM OF NEURONS"
		"FILTER SIZE, CHANNEL SIZE"
		"PARAMETERS & CHANNEL SIZE MEMORY"
		"RF & X_OUT formula"
		self.conv1 = nn.Conv2d(1, 32, 3, padding=1)      #(W)Filter Size= , (X_out)Channel Size=, RF: , (PARAMETERS & MEMORY, CHANNEL SIZE)"
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.pool1 = nn.MaxPool2d(2, 2)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.pool2 = nn.MaxPool2d(2, 2)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv5 = nn.Conv2d(256, 512, 3)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv6 = nn.Conv2d(512, 1024, 3)
		# r_in: , n_in: , j_in: , s: , r_out: , n_out: , j_out:
		self.conv7 = nn.Conv2d(1024, 10, 3)
# Correct values
# https://user-images.githubusercontent.com/498461/238034116-7db4cec0-7738-42df-8b67-afa971428d39.png
	def forward(self, x):
		x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
		x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
		x = F.relu(self.conv6(F.relu(self.conv5(x))))
		x = F.relu(self.conv7(x)) #<< Correct answer is to remove the ReLU from here
		x = x.view(-1, 10)
		return F.log_softmax(x)

#%%
model = FirstDNN()
summary(model, input_size=(1, 28, 28))

#%%
from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
	model.train()
	model.to(device)
	pbar = tqdm(train_loader)
	for batch_idx, (data, target) in enumerate(pbar):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		pbar.set_description("loss= %f,     batch_id= %d" % (loss.item(),batch_idx))
#		pbar.set_description("loss= {0},    batch_id= {1}   ".format(loss.item(),batch_idx))
#		pbar.set_description("loss= {0.2f}, batch_id= {0.2f}".format(loss.item(),batch_idx))
		
		
def test(model, device, test_loader):
	model.eval()
	test_loss_total = 0
	correct_preds_total = 0
	processed_total = 0
	pbar = tqdm(test_loader)
	with torch.no_grad():
		test_loss_batch = 0
		correct_preds_batch = 0
		for data, target in pbar:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss_batch = F.nll_loss(output, target, reduction='sum').item()   # sum up batch loss
			preds_batch = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct_preds_batch += pred.eq(target.view_as(preds_batch)).sum().item()
			
			test_loss_total += loss_batch
			correct_preds_total += correct_preds_batch
			

			count += 1
			pbar.set_description("batch loss = %f\t,batch correct = %d\t,batch accuracy %f",(test_loss,correct,target))
	
	test_loss_total /= count
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
		format(test_loss_total, correct, count,100. * correct / count))
	
#%%
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 2):
	train(model, device, train_dataloader, optimizer, epoch)
	test(model, device, test_dataloader)