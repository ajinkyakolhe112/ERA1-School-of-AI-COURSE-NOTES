from torch import nn as nn
import torch

class model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.container = {                      # RF Start = 1
            "conv1": nn.Conv2d(1,32,3,1,1),     # Delta RF = +2 
            "conv2": nn.Conv2d(32,64,3,1,1),    # Delta RF = +2
            "pool1": nn.MaxPool2d(2,2,0),       # Delta RF = *2
            "conv3": nn.Conv2d(64,128,3,1,1),   # Delta RF = +2
            "conv4": nn.Conv2d(128,256,3,1,1),  # Delta RF = +2
            "pool2": nn.MaxPool2d(2,2,0),       # Delta RF = *2. RF Total = 28
            "conv5": nn.Conv2d(256,512,3,1,0),  # Delta RF = +2
            "conv6": nn.Conv2d(512,512,3,1,0),  # Delta RF = +2, Output = (K:512,H:3,W:3)

        }
        self.sequential_model = nn.Sequential( *self.container.values() )

    def forward(self,x_train_batch):
        X = x_train_batch
        model = self.sequential_model

        result = model(X)
        result = self.sequential_model(x_train_batch)
        return result

def test_model():
    img = torch.randn(1,28,28)
    model = model_v1()

    output = model.sequential_model(img)
    output = model(img)
    "This structure allows for increamental testing layer by layer"
    output.shape

if __name__=="__main__":
    test_model()

    print("END")