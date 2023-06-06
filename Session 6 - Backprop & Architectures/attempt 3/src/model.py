from torch import nn as nn
import torch

class model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.container = {      # RF Start = 1
            1: nn.Conv2d,       # Delta RF = +2 
            2: nn.Conv2d,       # Delta RF = +2
            3: nn.MaxPool2d,    # Delta RF = *2
            4: nn.Conv2d,       # Delta RF = +2
            5: nn.Conv2d,       # Delta RF = +2
            6: nn.MaxPool2d,    # Delta RF = *2. RF Total = 28
            7: nn.Conv2d,       # Delta RF = +2
            8: nn.Conv2d,       # Delta RF = +2
            9: nn.Flatten,      # RF Total = 32

            10: nn.Linear,

            11: nn.Linear,
            12: nn.Softmax,

        }

    def forward(self,x_train_batch):
        X = x_train_batch
        return self.model(X)

def test_model():
    model = model_v1()
    model(torch.randn(1,28,28))

if __name__=="__main__":
    test_model()