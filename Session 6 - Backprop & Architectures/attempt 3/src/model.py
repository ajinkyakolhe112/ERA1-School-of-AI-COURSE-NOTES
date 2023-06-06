from torch import nn as nn
import torch

class model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential()

    def forward(self,x_train_batch):
        X = x_train_batch
        return self.model(X)
        result = 0
        return result

def test_model():
    model = model_v1()
    model(torch.randn(1,28,28))

if __name__=="__main__":
    test_model()