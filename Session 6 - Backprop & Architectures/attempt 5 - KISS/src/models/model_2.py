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
    # ! Target RF Reached: Have seen entire image now
	# ! BLOCK 4: OBJECT or CLASSIFICATION OBJECT for each pixel

	# "conv7": nn.Conv2d(32, 10, 3),

    "reshaper": reshape(1024*3*3),	# One Pixel for One Object from BLOCK 4
    "fc1": nn.Linear(1024*3*3,50),
	"fc2": nn.Linear(50,10),                # 10 Neurons for 10 Classes. Output = [1*10]
    "log_softmax": nn.LogSoftmax(dim=1),
}

model = nn.Sequential(*container.values())

if __name__=="__main__":
    test_img = torch.randn(3,1,28,28)
    output = new_model(test_img)
    output
    print("END")
