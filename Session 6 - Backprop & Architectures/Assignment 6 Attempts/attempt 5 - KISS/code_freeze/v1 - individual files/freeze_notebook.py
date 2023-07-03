#%%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchinfo import summary

BATCH_SIZE = 64
num_classes = 10
LR = 0.01
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data to plot accuracy and loss graphs
train_losses = [None]*NUM_EPOCHS
train_accuracy = [None]*NUM_EPOCHS

test_losses = [None]*NUM_EPOCHS
test_accuracy = [None]*NUM_EPOCHS

#%%
train_dataset = torchvision.datasets.MNIST(root = './data',train = True,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1307,), std = (0.3081,))]), download = True)
test_dataset = torchvision.datasets.MNIST(root = './data',train = False,transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1325,), std = (0.3105,))]),download=True)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = BATCH_SIZE,shuffle = True)

#%%
class reshape(nn.Module):
	def __init__(self, pixels):
		super().__init__()
		self.pixels = pixels
		
	def forward(self, x):
		x = x.view(-1, self.pixels)
		return x
	
container = {
	"conv1": nn.Conv2d(1, 32, (3, 3), 1, 1),
	"relu": nn.ReLU(),
	"conv2": nn.Conv2d(32, 32, (3, 3), 1, 1),
	"relu": nn.ReLU(),
	
	"pool1": nn.MaxPool2d((2, 2), 2, 0),
	"conv3": nn.Conv2d(32, 32, (3, 3), 1, 1),
	"relu": nn.ReLU(),
	"conv4": nn.Conv2d(32, 32, (3, 3), 1, 1),
	"relu": nn.ReLU(),
	"pool2": nn.MaxPool2d((2, 2), 2, 0),
	
	"conv5": nn.Conv2d(32, 32, (3, 3), 1, 0),
	"relu": nn.ReLU(),
	
	"conv6": nn.Conv2d(32, 32, (3, 3), 1, 0),
	"relu": nn.ReLU(),
	
	"conv7": nn.Conv2d(32, 10, 3),
	
	"reshaper": reshape(10),
	"log_softmax": nn.LogSoftmax(dim=1),
}

#%%
def train(train_loader, model, error_func, optimizer, epoch_no, device=None):
	
	model.to(device)
	pbar = tqdm(train_loader)
	
	train_loss_total = 0
	correct_preds_total = 0
	for b_id, (images, labels) in enumerate(pbar):
		images,labels = images.to(device), labels.to(device)
		
		#Forward pass
		outputs = model(images)
		error_batch = error_func(outputs, labels,reduction='mean')
		
		# Backward and optimize
		optimizer.zero_grad()
		error_batch.backward()
		optimizer.step()
		
		pred_labels = outputs.argmax(dim=1, keepdim=True).squeeze()
		correct_preds_batch = torch.eq(pred_labels, labels).sum().item()
		accuracy_batch = 100.0 * correct_preds_batch / BATCH_SIZE
		pbar.set_description("Epoch %d, Batch= %d, Batch Error Value = %f, Batch Pred Acc = %f" % (epoch+1,b_id+1, error_batch.item(), accuracy_batch))
		correct_preds_total = correct_preds_total + correct_preds_batch
		train_loss_total = error_batch.item()
		if (b_id + 1) % 100 == 0:
			print("Epoch %d, Batch %d, Error Value = %f, Pred_Acc = %f" % (epoch+1,b_id+1, error_batch.item(), accuracy_batch))
	train_losses[epoch_no] = train_loss_total / len(train_loader)
	train_accuracy[epoch_no] = 100.0 * correct_preds_total / len(train_loader)
	print("Epoch %d, Final Accuracy of the network %f" % (epoch +1 , train_accuracy[epoch_no]))

def test(test_loader, model, error_func,epoch, device=None):
	torch.set_grad_enabled(False)
	
	test_loss_total = 0
	correct_preds_total = 0
	pbar = tqdm(test_loader)
	for b_id,(images, labels) in enumerate(pbar):
		images,labels = images.to(device), labels.to(device)
		
		outputs = model(images)
		error_batch = error_func(outputs, labels, reduction='mean')
		
		pred_labels = outputs.argmax(dim=1, keepdim=True).squeeze()
		correct_preds = torch.eq(pred_labels, labels).sum().item()
		accuracy_batch = correct_preds / BATCH_SIZE * 100
		
		test_loss_total = test_loss_total + error_batch.item()
		correct_preds_total += correct_preds
		
		pbar.set_description("Epoch %d, Batch= %d, Error Value = %f, Pred Acc = %f" % (epoch+1, b_id+1, error_batch.item(), accuracy_batch))
		if (b_id + 1) % 100 == 0:
			print("Epoch %d, Batch %d, Error Value = %f, Pred_Acc = %f" % (epoch+1,b_id+1, error_batch.item(), accuracy_batch))
	test_losses[epoch] = test_loss_total / len(train_loader)
	test_accuracy[epoch] = 100.0 * correct_preds_total / len(test_loader)
	print("Epoch %d, Final Accuracy of the network %f" % (epoch +1 , 100.0 * correct_preds_total / len(test_loader)))
	torch.set_grad_enabled(True)
#%%
model = nn.Sequential(*container.values())
error_func = torch.nn.functional.nll_loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%%

for epoch in range(NUM_EPOCHS):
	print("epoch no= ",epoch+1)
	train(train_loader, model, error_func, optimizer,epoch)
	test(train_loader, model, error_func,epoch)

#%%
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_accuracy)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_accuracy)
axs[1, 1].set_title("Test Accuracy")