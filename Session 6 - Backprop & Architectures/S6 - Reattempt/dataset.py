import torch
import torchvision
import tensorflow as tf


train_dataset  = torchvision.datasets.MNIST(root=".data", download=True, train=True, transform=torchvision.transforms.ToTensor())
test_dataset   = torchvision.datasets.MNIST(root = ".data",download=True,train=False, transform = torchvision.transforms.ToTensor())

def test_dataset():
    print(train_dataset.shape,test_dataset.shape)
    print("END")