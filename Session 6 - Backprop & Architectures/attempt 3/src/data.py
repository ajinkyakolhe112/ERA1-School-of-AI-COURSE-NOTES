import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import logging

train_dataset = datasets.MNIST(".data",
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(".data",
                              train=False,
                              download=True,
                              transform=transforms.ToTensor())

train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=64,
                                               shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=64,
                                              shuffle=True)

def test():

    for batch_id,(x_train_batch,y_train_batch) in enumerate(train_dataloader):
        X,Y = x_train_batch, y_train_batch
        print("Batch = ",batch_id,"\t with X Shape = ",X.shape)


        if batch_id == 5:
            break

if __name__ =="__main__":
    test()