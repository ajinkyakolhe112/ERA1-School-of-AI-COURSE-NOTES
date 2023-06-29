import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # k:3, RF: 3 | Output: 28 -> 28 | Channels: (1 -> 32)
        # Layer 1: Parameters = 32 * (1 * 3*3)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)

        # k:3, RF: 5 | Output: 28 -> 28 | Channels: (32 -> 64)
        # Layer 2: Parameters = 64 * (32 * 3*3)   
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # k:2, RF: 10 | Output: 28 -> 14 | Channels: (64 -> 64)
        # Layer 3: Parameters = 0
        self.pool1 = nn.MaxPool2d(2, 2)

        # k:3, RF: 12 | Output: 14 -> 14 | Channels: (64 -> 128)
        # Layer 4: Parameters = 128 * (64 * 3*3)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # k:3, RF: 14 | Output: 14 -> 14 | Channels: (128 -> 256)
        # Layer 5: Parameters = 256 * (128 * 3*3)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        # k:2, RF: 28 | Output: 14 -> 7 | Channels: (256 -> 256)
        # Layer 6: Parameters = 0
        self.pool2 = nn.MaxPool2d(2, 2)

        # k:3, RF: 30 | Output:  7 -> 5 | Channels: (256 -> 512)
        # Layer 7: Parameters = 512 * (256 * 3*3)           
        self.conv5 = nn.Conv2d(256, 512, 3)

        # k:3, RF: 32 | Output:  5 -> 3 | Channels: (512 -> 1024)
        # Layer 8: Parameters = 1024 * (512 * 3*3)        
        self.conv6 = nn.Conv2d(512, 1024, 3)

        # k:3, RF: 34 | Output:  3 -> 1 | Channels (1 -> 32)    
        # Layer 9: Parameters = 1024 * (10 * 3*3)      
        self.conv7 = nn.Conv2d(1024, 10, 3)             

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)

        # 1*1*10 > (-1,10)
        x = x.view(-1, 10)                              
        # -1 means last dimention which is dim 1 in this case
        # Two dimentions. dim 0 & dim 1
        output = F.log_softmax(x, dim= 1) # OR F.log_softmax(x, dim=-1)
        probs = F.softmax(x,dim=1)
        
        return output
