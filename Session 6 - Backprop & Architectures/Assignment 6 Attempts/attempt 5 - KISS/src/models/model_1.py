import torch.nn as nn

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


def get_model_1(**kwargs):
	model_1 = nn.Sequential( * model_1_init.values() ) 
	# list of Values in Dictory, is passed as Stack of Layers to nn.Sequential by using *
	
	return model_1

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [1, 10]                   --
├─Conv2d: 1-1                            [32, 28, 28]              320
│    └─weight                                                      ├─288
│    └─bias                                                        └─32
├─ReLU: 1-2                              [32, 28, 28]              --
├─Conv2d: 1-3                            [32, 28, 28]              9,248
│    └─weight                                                      ├─9,216
│    └─bias                                                        └─32
├─MaxPool2d: 1-4                         [32, 14, 14]              --
├─Conv2d: 1-5                            [32, 14, 14]              9,248
│    └─weight                                                      ├─9,216
│    └─bias                                                        └─32
├─Conv2d: 1-6                            [32, 14, 14]              9,248
│    └─weight                                                      ├─9,216
│    └─bias                                                        └─32
├─MaxPool2d: 1-7                         [32, 7, 7]                --
├─Conv2d: 1-8                            [32, 5, 5]                9,248
│    └─weight                                                      ├─9,216
│    └─bias                                                        └─32
├─Conv2d: 1-9                            [32, 3, 3]                9,248
│    └─weight                                                      ├─9,216
│    └─bias                                                        └─32
├─Conv2d: 1-10                           [10, 1, 1]                2,890
│    └─weight                                                      ├─2,880
│    └─bias                                                        └─10
├─reshape: 1-11                          [1, 10]                   --
├─LogSoftmax: 1-12                       [1, 10]                   --
==========================================================================================
Total params: 49,450
Trainable params: 49,450
Non-trainable params: 0
Total mult-adds (M): 19.26
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.51
Params size (MB): 0.20
Estimated Total Size (MB): 0.71

"""
