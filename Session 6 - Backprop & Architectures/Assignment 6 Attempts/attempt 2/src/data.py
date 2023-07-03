from torchvision import datasets,transforms
import torch
import torch.nn as nn
from tqdm import tqdm

trainData = datasets.MNIST(train=True,download=True,root="./.data")
testData = datasets.MNIST(train=False,download=True,root="./.data")

"Transforms on PIL Image only"
transformPipeline = transforms.Compose(
    [
    transforms.ToTensor()
    ]
)

trainDataT = datasets.MNIST(train=True,download=False,root="./.data",transform=transformPipeline) # transform vs transforms
testDataT = datasets.MNIST(train=False,download=False,root="./.data",transform=transformPipeline)

trainDataLoader = torch.utils.data.DataLoader(trainDataT,batch_size=32,shuffle=True)
testDataLoader = torch.utils.data.DataLoader(testDataT,batch_size=32,shuffle=True)


if __name__=="__main__":
    "Error before torch 2.0"
    "Compose transform does not support torchscript"
    try:
        testTransform = nn.Sequential(transforms.ToTensor())
        testTransform(torch.randn(1,28,28))
    except Exception as E:
        print(E)
    
    pbar = tqdm(trainDataLoader)
    for batch_id,(Xdata,Yactual) in enumerate(pbar):
        pbar.set_description("(batch=%d),\t"%batch_id)


print("END")