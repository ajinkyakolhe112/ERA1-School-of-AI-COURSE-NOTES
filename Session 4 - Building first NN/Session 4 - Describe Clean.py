"CODE BLOCK 1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#%%
"CODE BLOCK 2"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device

"CODE BLOCK 3"
# Train data transformations
"Same Train & Test Transformation"
train_transforms = transforms.Compose([
#	transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
#	transforms.Resize((28, 28)),
#	transforms.RandomRotation((-15., 15.), fill=0),
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
	])

# Test data transformations
test_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
	])

"CODE BLOCK 4"
# Two mistakes
"train=False & transform = test_transforms"
train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

"CODE BLOCK 5"
#%%
"Shuffle = True for Data Batch & test_data loader"
batch_size = 512
kwargs = {'batch_size': batch_size, 'shuffle': True, 'pin_memory': True} #FixME: spec error on coderunner 4. works fine on colab
test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
train_loader = torch.utils.data.DataLoader(train_data, **kwargs)

"CODE BLOCK 6"
#%%
"""
import matplotlib.pyplot as plt
batch_data, batch_label = next(iter(train_loader)) # First batch is being discarded here. 
fig = plt.figure()
for i in range(12):
	plt.subplot(3,4,i+1)
	plt.tight_layout()
	plt.imshow(batch_data[i].squeeze(0), cmap='gray')
	plt.title(batch_label[i].item())
	plt.xticks([])
	plt.yticks([])
"""
	
"CODE BLOCK 7"
#%%
class Net(nn.Module):
	#This defines the structure of the NN.
	def __init__(self):
		super(Net, self).__init__()
		# 4 blocks of 4 conv layers each extracting increasing in number & higher depth features each
		# 2 max pooling & 4 conv layers. (image size after: (28 - 4*2)/2*2
		self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # 26*26
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # 24*24, 12*12
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3) # 10*10
		self.conv4 = nn.Conv2d(128, 256, kernel_size=3) # 8*8, 4*4
		# output Image = (256,5,5)
		# check transform stage in forward
		self.fc1 = nn.Linear(256*4*4, 50) # singleImage = 256 Channels. Size = 4*4. Converted to Vector. Output = (1*50) or (1,50)
		self.fc2 = nn.Linear(50, 10) # Output = (1*10) or (1,10)
		
	def forward(self, x):
		x = F.relu(self.conv1(x), 2) # 2 here is inplace=False. 
		x = F.relu(F.max_pool2d(self.conv2(x), 2)) # 2 = Kernel Size here. 
		x = F.relu(self.conv3(x), 2)
		x = F.relu(F.max_pool2d(self.conv4(x), 2)) 
		
		# output Image = (-1,256,5,5)
		# incoming vector = (-1,1,Multiplication)
		x = x.view(-1, 256*4*4)
		x = F.relu(self.fc1(x)) # RELU? Extracted features are being discarded may be?
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1) # F.softmax(x)[0].sum()
		return x

testModel = Net()
imageBatch = torch.randn(512,1,28,28)
testModel(imageBatch)

#%%
# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

#%%
"BLOCK 8"
from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
	return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer):
	model.train()
	train_loader = tqdm(train_loader)
	
	train_loss = 0
	correct = 0
	processed = 0
	
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		pred = model(data)
		loss = F.nll_loss(pred, target)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		train_loss+=loss.item()
		correct += GetCorrectPredCount(pred, target)
		processed += len(data)
		train_loader.set_description("Train: Loss=%f, Batch_id=%0.2f, Accuracy=%0.4f"%(loss.item(),batch_idx,100*correct/processed))
	
	train_acc.append(100*correct/processed)
	train_losses.append(train_loss/len(train_loader))
	
def test(model, device, test_loader):
	model.eval()

	test_loss = 0
	correct = 0

	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = F.nll_loss
			
			test_loss += loss(output, target, reduction='sum').item()  # sum up batch loss
			correct += GetCorrectPredCount(output, target)

	test_loss /= len(test_loader.dataset)
	test_acc.append(100. * correct / len(test_loader.dataset))
	test_losses.append(test_loss)

	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.
		format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

"BLOCK 9"
#%%
model = Net()
model.to(device) # Model, Data both needs to be on device
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)

num_epochs = 20

for epoch in range(1, num_epochs+1):
	print(f'Epoch {epoch}')
	train(model, device, train_loader, optimizer)
	scheduler.step()
	
test(model, device, test_loader) # Test Loader
	
#%%
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

#%%
#!pip install torchsummary
from torchinfo import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model,input_size=(1,28,28),verbose=2,
	col_names=["input_size","kernel_size", "output_size", "num_params", "params_percent"],col_width=20);

print("End")