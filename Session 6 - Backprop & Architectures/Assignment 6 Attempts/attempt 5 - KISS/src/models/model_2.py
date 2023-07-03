import torch
import torch.nn as nn
from torchinfo import summary

class reshape(nn.Module):
    def __init__(self, pixels):
        super().__init__()
        self.pixels = pixels

    def forward(self, x):
        x = x.view(-1, self.pixels)
        return x

container = {
    # ! RF Start = 1
    # Format: img = [ in_channels, out_channels ] , kernel = [ size, stride, padding ]
	
	"conv0": nn.Conv2d(1,16,	(3,3), 1, 0),

	# BLOCK 1 - TODO: Extract Features in Channels & Compress Channels with 1*1
	"Batch Norm 1": nn.BatchNorm2d(16),
    "conv1": nn.Conv2d(16, 32, 	(3, 3), 1, 1),
    "relu": nn.ReLU(),
	"Batch Norm 2": nn.BatchNorm2d(32),
    "conv2": nn.Conv2d(32, 64, 	(3, 3), 1, 1),
    "relu": nn.ReLU(),

    "pool1": nn.MaxPool2d((2, 2), 2, 0),
	"compress_block1": nn.Conv2d(64,16,	(1,1), 1, 0), 

	# BLOCK 2
	"Batch Norm 3": nn.BatchNorm2d(16),
    "conv3": nn.Conv2d(16, 32, 	(3, 3), 1, 1),
    "relu": nn.ReLU(),

	"Batch Norm 4": nn.BatchNorm2d(32),
    "conv4": nn.Conv2d(32, 64, 	(3, 3), 1, 1),
    "relu": nn.ReLU(),

    "pool2": nn.MaxPool2d((2, 2), 2, 0),
	"compress_block2": nn.Conv2d(64,16,	(1,1), 1, 0), 

	"Batch Norm 5": nn.BatchNorm2d(16),
	# BLOCK 3
    "conv5": nn.Conv2d(16, 32, 	(3, 3), 1, 0),
    "relu": nn.ReLU(),
	# "compress_block3": nn.Conv2d(32, 32,	(1,1), 1, 0), 
    
	# BLOCK 4
	"Batch Norm 6": nn.BatchNorm2d(32),
    "conv6": nn.Conv2d(32, 64, (3, 3), 1, 0),    # ! Output = (K:1024, H:3, W:3)
    "relu": nn.ReLU(),
	"compress_block4": nn.Conv2d(64, 16, (1,1), 1, 0),
    # ! Target RF Reached: Have seen entire image now
	# ! BLOCK 4: OBJECT or CLASSIFICATION OBJECT for each pixel
	
	# output: torch.Size([3, 64, 2, 2])
	
	# "conv7": nn.Conv2d(64, 20, 3),
	
	"GAP": nn.AvgPool2d(2,2),

    "reshaper": reshape(16),	# One Pixel for One Object from BLOCK 4
    "fc1": nn.Linear(16,50),
	"fc2": nn.Linear(50,10),                # 10 Neurons for 10 Classes. Output = [1*10]
    "log_softmax": nn.LogSoftmax(dim=1),
}

class customSequential(nn.Module):
	def __init__(self, container):
		super().__init__()
		self.container = container
		self.internal_sequential_model = nn.Sequential( * container.values())
		
	def forward(self,input_x):
		self.container
		self.internal_sequential_model

		for name,layer in self.container.items():
			# print(name,layer)
			output = layer(input_x)
			# print(name, output.shape)
			input_x = output
			output
		return output

def get_model_2():
	model = customSequential(container)
	return model
"""
Good to imagine on Imagenet. Real life objects are easier to visualize.
Imagine a classification problem.
	nn.Sequential( Successive distillation. 
		nn.Conv2d(400, (7,7)), # Localized intelligence. Looks at image, and extracts only information edges & gradients. Or adds edges & gradients info. Extract into channels & combine clustering
		nn.MaxPool2d(2,2)
		nn.Conv2d(400,(7,7)),  # Previous condensed info as input, and further advanced processing
		nn.Conv2d(400,feature maps, (7,7)), # Each Neuron Expected to create, parts of object. How many parts of objects do we have. (Nose of dog). Huge number of sub features
		nn.Conv2d(400 feature maps,(7,7)), 1*1. Each neuron should be, a object relevent to our classes. Dog # imagenet, 10k objects at least. Each with
		nn.Conv2d(10, (1,1)) # Should give us, 10 images we want.
		nn.Linear(10,10),
		nn.Linear(10,10) # 
	)
"""

if __name__=="__main__":
	test_img = torch.randn(3,1,28,28)
	# model = nn.Sequential(* container.values() )

	model = customSequential(container)
	output = model(test_img)

	summary(model.internal_sequential_model, input_size=(1, 1, 28, 28), verbose=2);

	output = model(test_img)
	output
	print("END")

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [1, 10]                   --
├─Conv2d: 1-1                            [1, 16, 26, 26]           160
│    └─weight                                                      ├─144
│    └─bias                                                        └─16
├─BatchNorm2d: 1-2                       [1, 16, 26, 26]           32
│    └─weight                                                      ├─16
│    └─bias                                                        └─16
├─Conv2d: 1-3                            [1, 32, 26, 26]           4,640
│    └─weight                                                      ├─4,608
│    └─bias                                                        └─32
├─ReLU: 1-4                              [1, 32, 26, 26]           --
├─BatchNorm2d: 1-5                       [1, 32, 26, 26]           64
│    └─weight                                                      ├─32
│    └─bias                                                        └─32
├─Conv2d: 1-6                            [1, 64, 26, 26]           18,496
│    └─weight                                                      ├─18,432
│    └─bias                                                        └─64
├─MaxPool2d: 1-7                         [1, 64, 13, 13]           --
├─Conv2d: 1-8                            [1, 16, 13, 13]           1,040
│    └─weight                                                      ├─1,024
│    └─bias                                                        └─16
├─BatchNorm2d: 1-9                       [1, 16, 13, 13]           32
│    └─weight                                                      ├─16
│    └─bias                                                        └─16
├─Conv2d: 1-10                           [1, 32, 13, 13]           4,640
│    └─weight                                                      ├─4,608
│    └─bias                                                        └─32
├─BatchNorm2d: 1-11                      [1, 32, 13, 13]           64
│    └─weight                                                      ├─32
│    └─bias                                                        └─32
├─Conv2d: 1-12                           [1, 64, 13, 13]           18,496
│    └─weight                                                      ├─18,432
│    └─bias                                                        └─64
├─MaxPool2d: 1-13                        [1, 64, 6, 6]             --
├─Conv2d: 1-14                           [1, 16, 6, 6]             1,040
│    └─weight                                                      ├─1,024
│    └─bias                                                        └─16
├─BatchNorm2d: 1-15                      [1, 16, 6, 6]             32
│    └─weight                                                      ├─16
│    └─bias                                                        └─16
├─Conv2d: 1-16                           [1, 32, 4, 4]             4,640
│    └─weight                                                      ├─4,608
│    └─bias                                                        └─32
├─BatchNorm2d: 1-17                      [1, 32, 4, 4]             64
│    └─weight                                                      ├─32
│    └─bias                                                        └─32
├─Conv2d: 1-18                           [1, 64, 2, 2]             18,496
│    └─weight                                                      ├─18,432
│    └─bias                                                        └─64
├─Conv2d: 1-19                           [1, 16, 2, 2]             1,040
│    └─weight                                                      ├─1,024
│    └─bias                                                        └─16
├─AvgPool2d: 1-20                        [1, 16, 1, 1]             --
├─reshape: 1-21                          [1, 16]                   --
├─Linear: 1-22                           [1, 50]                   850
│    └─weight                                                      ├─800
│    └─bias                                                        └─50
├─Linear: 1-23                           [1, 10]                   510
│    └─weight                                                      ├─500
│    └─bias                                                        └─10
├─LogSoftmax: 1-24                       [1, 10]                   --
==========================================================================================
Total params: 74,336
Trainable params: 74,336
Non-trainable params: 0
Total mult-adds (M): 20.03
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 1.10
Params size (MB): 0.30
Estimated Total Size (MB): 1.40
==========================================================================================

"""
