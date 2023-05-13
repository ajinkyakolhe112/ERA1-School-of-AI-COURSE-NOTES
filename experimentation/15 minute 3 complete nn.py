import torch
import torch.nn as nn
from torchinfo import summary

from torchvision import datasets,transforms

## Model
class customNN(nn.Module):
	def __init__(self):
		super().__init__()
		self.convLayer1 = nn.Conv2d(1,10,28) # (X_in) (Ch_in*28*28) * (W)(Ch_in*21*21) * 10 = (X_out)10*8*8
	
	def forward(self,input):
		transform = self.convLayer1(input)
		transform = nn.functional.relu(transform)
		
		# reduced output size according to padding & stride
		transform = transform.view(-1,10) # Flatten to 2D, easy resize to 10 because 10 classes.
		finalResult = nn.functional.log_softmax(transform)
		return finalResult

model = customNN()

vars(model)
dir(model)
model
type(model)
id(model)

testImage = torch.randn(1,28,28)
model.forward(testImage)
output = model(testImage)
model.convLayer1(testImage)
model.convLayer1.forward(testImage) 

## DataLoader
train_dataloader = torch.utils.data.DataLoader(
	datasets.MNIST("./data",download=True,train=True,
		transform = transforms.ToTensor()
	),
	batch_size=128,shuffle=True 
)
test_dataloader = torch.utils.data.DataLoader(
	datasets.MNIST("./data",download=True,train=False,
		transform = transforms.ToTensor()
	),
	batch_size=128,shuffle=True
)

for i, (inputData, yActual) in enumerate(train_dataloader):
	print(i,inputData,yActual)
	test = 0
	break

## Training. Error, Error Contribution, Weight Correction according to contribution
inputDataX,yActualX = next(train_dataloader)
yPredictedX = model(inputDataX)
convexErrorFunction = nn.functional.nll_loss(yActualX,yPredictedX)

convexErrorFunction.backward() # FROM HERE CONTINUE TOMORROW


# to get a working model, need to fiddle with matrix sizes to get it right.


# Most minimalist image model would be each kernel/feature for its final predicted value 1/0. A widest nn. Each nn can be put as a wide nn. it means, the number of features is a good measure of complexity of the problem. 

## Getting to Know PARAMETERS, size in memory of gpu. Number of matrix multiplications
## Getting to know feature spaces
## These codes are precise instructions. It runs only if its exactly correct. 
## Operations are matrix multiply, and vector addition and a few simple maths functions. and error calculation. Hence, we need to completely understand matrix dimentions which are formed when we build layers.
## Custom Defined terms. DIKW. Knowledge Density., Problem Complexity , Understanding Channels & Photoshop Layers analogy
## Model Capacity

print("END")
