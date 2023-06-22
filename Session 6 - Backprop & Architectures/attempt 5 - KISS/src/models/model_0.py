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
    "conv1": nn.Conv2d(1, 32, (3, 3), 1, 1),
    "relu": nn.ReLU(),
    "conv2": nn.Conv2d(32, 64, (3, 3), 1, 1),
    "relu": nn.ReLU(),
    "pool1": nn.MaxPool2d((2, 2), 2, 0),

    "conv3": nn.Conv2d(64, 128, (3, 3), 1, 1),
    "relu": nn.ReLU(),
    "conv4": nn.Conv2d(128, 256, (3, 3), 1, 1),
    "relu": nn.ReLU(),
    "pool2": nn.MaxPool2d((2, 2), 2, 0),

    "conv5": nn.Conv2d(256, 512, (3, 3), 1, 0),
    "relu": nn.ReLU(),
    
    "conv6": nn.Conv2d(512, 1024, (3, 3), 1, 0),    # ! Output = (K:1024, H:3, W:3)
    "relu": nn.ReLU(),
    # ! Target RF Reached: All Features Extracted By Now

	# "conv7": nn.Conv2d(32, 10, 3),

    "reshaper": reshape(1024*3*3),
    "fc1": nn.Linear(1024*3*3,50),
	"fc2": nn.Linear(50,10),                # 10 Neurons for 10 Classes. Output = [1*10]
    "log_softmax": nn.LogSoftmax(dim=1),
}

def get_model_0():
    model = nn.Sequential(*container.values())

    return model

"""
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [1, 10]                   --
├─Conv2d: 1-1                            [32, 28, 28]              320
│    └─weight                                                      ├─288
│    └─bias                                                        └─32
├─ReLU: 1-2                              [32, 28, 28]              --
├─Conv2d: 1-3                            [64, 28, 28]              18,496
│    └─weight                                                      ├─18,432
│    └─bias                                                        └─64
├─MaxPool2d: 1-4                         [64, 14, 14]              --
├─Conv2d: 1-5                            [128, 14, 14]             73,856
│    └─weight                                                      ├─73,728
│    └─bias                                                        └─128
├─Conv2d: 1-6                            [256, 14, 14]             295,168
│    └─weight                                                      ├─294,912
│    └─bias                                                        └─256
├─MaxPool2d: 1-7                         [256, 7, 7]               --
├─Conv2d: 1-8                            [512, 5, 5]               1,180,160
│    └─weight                                                      ├─1,179,648
│    └─bias                                                        └─512
├─Conv2d: 1-9                            [1024, 3, 3]              4,719,616
│    └─weight                                                      ├─4,718,592
│    └─bias                                                        └─1,024
├─reshape: 1-10                          [1, 9216]                 --
├─Linear: 1-11                           [1, 50]                   460,850
│    └─weight                                                      ├─460,800
│    └─bias                                                        └─50
├─Linear: 1-12                           [1, 10]                   510
│    └─weight                                                      ├─500
│    └─bias                                                        └─10
├─LogSoftmax: 1-13                       [1, 10]                   --
==========================================================================================
Total params: 6,748,976
Trainable params: 6,748,976
Non-trainable params: 0
Total mult-adds (G): 18.74
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 1.38
Params size (MB): 27.00
Estimated Total Size (MB): 28.38
==========================================================================================
"""
if __name__=="__main__":
    test_img = torch.randn(3,1,28,28)
    output = new_model(test_img)
    output
    print("END")
