#%%
import torch
import torch.nn as nn
from torchvision import datasets, transforms

#%%
# Model for MNIST

sequentialModel = nn.Sequential(
#	SEQUENCE of transforms here. 4 Blocks. 
#	(First two blocks, enough to figure out all subfeatures. 
	nn.Conv2d(1,1,3),
	nn.ReLU(),
	
	nn.Conv2d(1,1,3),
	nn.ReLU(),
	
#	Last 2 blocks, making decision from distilled features from those two blocks)
	nn.Conv2d(1,1,3),
	nn.ReLU(),
	
	nn.Conv2d(1,10,3),
	#nn.LogSoftmax(10)
)
testImage = torch.randn(1,28,28)
sequentialModel(testImage)

type(sequentialModel)
vars(sequentialModel)
dir(sequentialModel)

#%%
class dynamicModel(nn.Module):
	def __init__(self):
		#super.__init__() 			# initialize nn.Module & everything needed for this model
		super().__init__()
#		initalize same transform layers. (extending nn.Module)
#		classes need to be initialized. for Activation, they don't hold weights so can directly call in forward.
		self.block1_conv1 = nn.Conv2d(1,1,3),
		self.block2_conv1 = nn.Conv2d(1,1,3),
		self.block3_conv1 = nn.Conv2d(1,1,3),
		self.block4_conv1 = nn.Conv2d(1,1,3),
		
		
		pass
	
	def forward(self, input):
#		SAME SEQUENCE of transform goes here
		block1_transform = self.block1_conv1(input)
		block1_transform = torch.nn.functional.ReLU(block1_transform)

		block2_transform = self.block2_conv1(block1_transform)
		block2_transform = torch.nn.functional.ReLU(block2_transform)
		
		block3_transform = self.block3_conv1(block2_transform)
		block3_transform = torch.nn.functional.ReLU(block3_transform)
		
		block2_transform = self.block4_conv1(block3_transform)
		#torch.nn.functional.log_softmax(block2_transform, 10)
		
		finalResult = block2_transform
		
		return finalResult

modelInstance = dynamicModel()
type(modelInstance)
vars(modelInstance) # Check vars of model vs vars of Sequential

testImage = torch.randn(1,28,28)
modelInstance(testImage)
#			modelInstance.forward(testImage)




print(modelInstance)
