import torch
import torch.nn as nn
from torchvision import datasets, transforms

class CustomModule(nn.Module):
	def __init__(self):
		super().__init__()  # Very important. Otherwise CustomModule won't be registered as NN
		self.conv = nn.Conv2d(1,1,3)
		
	def forward(self,input):
		result = self.conv(input)
		return result

model = CustomModule()
model.conv.weight,model.conv.bias

testImage = torch.randn(1,28,28)
model(testImage)
model.forward(testImage)

print(model)


