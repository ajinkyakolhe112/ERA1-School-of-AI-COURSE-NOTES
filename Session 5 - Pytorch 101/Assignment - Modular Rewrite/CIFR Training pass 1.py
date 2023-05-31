import torch
import torch.nn as nn
from torchinfo import summary
from torchvision import datasets, transforms
import torchvision
from tqdm import tqdm
import torch.nn.functional as F

todo = {
    "1": "DataSet, DataLoader",
    "2": "Model Architecture & Parameters Output Calculation",
    "3": "Forward Pass Manual Calculation Pass & Matmul Add Evaluation",
    "4": "Model Training Pass = Error Reduction via Parameter Update",
    "5": "Model Entire End to End training & Loss Evaluation",
    "6": "Experimenting with architecture to improve it iteratively",
}

"data is generally scarce in DL. So we expand it by edit original images to create fake new images. Increases training Data Size"
trainData = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)
testData = torchvision.datasets.CIFAR10(
    "./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    ),
)

"Data Loader is a light simple data loader function for getting data in batches. It gives data directly to model, hence its in numerically faster optmized Tensor format"
trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size=8, shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testData, batch_size=8, shuffle=True)

for batchId, (data, target) in enumerate(trainDataLoader):
    print(batchId, data.shape, target.shape)
    break
    pass

"Image = (3,32,32)"
"Classes: 10 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']"
class baseModel(torch.nn.Module):
    
    "Initialization of Transform Layers & Coefficients being used in Transforms"
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, (9, 9)),  # ( 32 -> 64 -> 256 -> 512 or
            nn.Conv2d(32, 64, (9, 9)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (9, 9)),
            nn.Conv2d(128, 256, (8, 8)),  # 256,1,1 (kernel size = image/4 blocks
            nn.ReLU(),
            nn.Flatten(),  # 2d Image to 1d Tensor. reshape, flatten, view
            nn.Linear(256, 1),
        )

    "Initialization of Transform Layers & Coefficients being used in Transforms"
    def forward(self, inputData):
        transformedData = self.conv1(inputData)
        transformedData

        pass


model = baseModel()
model(torch.randn(3, 3, 32, 32))

print("End")


"""


"""
