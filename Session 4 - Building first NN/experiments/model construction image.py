import torch
import torch.nn as nn

from torchinfo import summary
from torchvision import datasets,transforms
from sklearn import datasets as toysets



class reshape2d(nn.Module):
	def __init__(self,nClasses):
		super().__init__()
		self.nClasses = nClasses
	def forward(self,inputX): # inputX = C * H * W
		outputX = inputX.view(-1,self.nClasses) # n_dim = 1 ( channels, 1 dim. or channels, 2 dim, or channels , 3 dims
		return outputX


# LINEAR => Vector to Vector.. columns or features
# CONV2D ( chw

model = nn.Sequential( # X_{out} = X_{in} \odot W + b. X_{in} or X_{out} = C * H * W
	nn.Conv2d(in_channels = 1,out_channels = 100,kernel_size = (3,3)), # Important terms, channels in features & receptive fields & number of neurons
	nn.ReLU(),
	nn.Conv2d(in_channels = 100,out_channels = 200,kernel_size = (3,3)),
	nn.ReLU(),
	nn.Conv2d(in_channels = 200, out_channels = 50, kernel_size = (3,3)),
	nn.MaxPool2d((2,2),1),
	nn.ReLU(),
	nn.Conv2d(in_channels = 50,out_channels = 20,kernel_size = (3,3)),
	nn.ReLU(),
	nn.Conv2d(in_channels = 20,out_channels = 10,kernel_size = (3,3)),
	nn.MaxPool2d((2,2),1),
	nn.ReLU(),
	reshape2d(10),
	#nn.Linear(256,10),
	#nn.LogSoftmax(1),
)
testImage = torch.randn(1,28,28)
model(testImage)
print("End")


# y = X*W + b
class customNN(nn.Module):
	def __init__(self,conv_layers):
		super.__init__()
		nn.Conv2d(1,50,(3,3)),
		nn.Conv2d(50,100,(3,3)),
		nn.Conv2d(100,200,(3,3)),
		nn.Conv2d()
		
	def forward(self,inputX):
		
		finalOutput = 0
		return finalOutput



print("End")