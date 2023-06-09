import torch.nn as nn

def get_model(**kwargs):

	class reshape(nn.Module):
		def __init__(self, pixels):
			super().__init__()
			self.pixels = pixels
			
		def forward(self, x):
			x = x.view(-1, self.pixels)
			return x

	model_1_init = {
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

	model_1 = nn.Sequential( * model_1_init.values() ) # list of Values in Dictory, is passed as Stack of Layers to nn.Sequential by using *

	return model_1