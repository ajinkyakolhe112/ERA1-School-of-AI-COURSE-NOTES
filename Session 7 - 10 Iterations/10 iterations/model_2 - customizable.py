import torch.nn as nn
import torch.nn.functional as F

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Block 1
        self.block1 = nn.ModuleDict({
            "conv1": nn.Conv2d(1, 32, 3, padding=1),
            "relu" : nn.ReLU(),
            "conv2": nn.Conv2d(32, 64, 3, padding=1),
            "relu" : nn.ReLU(),
        })

        self.transition1 = nn.ModuleDict({
            "pool1": nn.MaxPool2d(2, 2),
        })

        # Block 2
        self.block2 = nn.ModuleDict({
            "conv3": nn.Conv2d(64, 128, 3, padding=1),
            "relu" : nn.ReLU(),
            "conv4": nn.Conv2d(128, 256, 3, padding=1),
            "relu" : nn.ReLU(),
        })
        
        self.transition2 = nn.ModuleDict({
            "pool1": nn.MaxPool2d(2, 2),
        })

        # Block 3
        self.block3 = nn.ModuleDict({
            "conv5": nn.Conv2d(256, 512, 3),
            "relu" : nn.ReLU(),
            "conv6": nn.Conv2d(512, 1024, 3),
            "relu" : nn.ReLU(),
        })

        # Block 4
        self.block4 = nn.ModuleDict({
            "conv7": nn.Conv2d(1024, 10, 3),
        })

    def forward(self, x):
        x = self.block1(x)
        x = self.transition1(x)
        
        x = self.block2(x)
        x = self.transition2(x)
        
        x = self.block3(x)

        x = self.block4(x)

        # (-1 = dim 0, 10 = dim 1)
        x = x.view(-1, 10)

        output = F.log_softmax(x, dim=1)
        probs = F.softmax(x, dim=1)
        return output
