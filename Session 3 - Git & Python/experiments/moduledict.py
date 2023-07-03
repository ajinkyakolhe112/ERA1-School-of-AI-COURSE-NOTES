import torch
import torch.nn as nn

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(1,32, 3, padding=1)    # 28 -> 28 | 3
        block = {
            "conv1": nn.Conv2d(32, 64, 3, padding=1),
            "relu1" : nn.ReLU(),
            "conv2": nn.Conv2d(64, 128, 3, padding=1),
            "relu2" : nn.ReLU(),
        }
        # Block 1
        self.block1 = nn.ModuleDict(block)

        # Maxpooling before or after 1x1 convolution?
        self.transition1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128,32,1), # Squeeze
        )

        # Block 2
        self.block2 = nn.ModuleDict(block)
        
        self.transition2 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128,32,1), # Squeeze
        )

        # Block 3
        self.block3 = nn.ModuleDict(block)

        self.transition3 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128,32,1), # Squeeze
        )

        # Block 4
        self.block4 = nn.ModuleDict({
            "conv7": nn.Conv2d(32, 10, 3),
        })

    def forward(self, x):
        b1, b2, b3, b4 = self.block1, self.block2, self.block3, self.block4

        x = self.conv0(x)

        x = b1.relu2(b1.conv2(b1.relu1(b1.conv1(x))))
        x = self.transition1(x)
        
        x = b2.relu2(b2.conv2(b2.relu1(b2.conv1(x))))
        x = self.transition2(x)
        
        x = b3.relu2(b3.conv2(b3.relu1(b3.conv1(x))))
        x = self.transition3(x)

        x = b4.conv7(x)

        # (-1 = dim 0, 10 = dim 1)
        x = x.view(-1, 10)

        output = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)
        return output


# When block was a container, and same container was used to create 4 module dict, then layers weren't being populated in function model_2.named_parameters()
# Duplicate names of layers, leads to them not being populated here. But state_dict() works fine. 
# VERY WRONG FOR THE MODEL
for name,param in model_2.named_parameters():
    if param.ndim > 1 :
        kernel = param.shape
        ch_out, ch_in, height, width = kernel
        print(name,ch_out * ch_in * height * width)