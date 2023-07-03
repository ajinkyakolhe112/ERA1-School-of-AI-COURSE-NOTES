from torchvision import datasets, transforms
import torchvision
import torch
"X"

def getDataLoader(batch_size=8):
    trainData = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,
        transform=transforms.Compose([transforms.ToTensor(),]),)
    
    testData = torchvision.datasets.CIFAR10("./data",train=False,download=True,
        transform=transforms.Compose([transforms.ToTensor(),]),)
    
    trainDataLoader = torch.utils.data.DataLoader(trainData, batch_size, shuffle=True)
    testDataLoader = torch.utils.data.DataLoader(testData, batch_size, shuffle=True)

    for batchId, (data, target) in enumerate(trainDataLoader):
        print(batchId, data.shape, target.shape)
        break
    
    return trainDataLoader,testDataLoader


#TODO: Should view the batch here. To be called when watching training of a few batches and testing of a few batches

details = "Image Size = (3,32,32) C*H*W \
Classes: 10 ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']"

print("End")
