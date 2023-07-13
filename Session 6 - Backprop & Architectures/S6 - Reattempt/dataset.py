import torch
import torchvision
import tensorflow as tf


train_dataset  = torchvision.datasets.MNIST(root=".data", download=True, train=True, transform= torchvision.transforms.ToTensor())
validation_dataset   = torchvision.datasets.MNIST(root = ".data", download=True, train=False, transform = torchvision.transforms.ToTensor())

keyword_value_dict = {
    "batch_size": 64,
}
train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, **keyword_value_dict)
test_loader = torch.utils.data.DataLoader(validation_dataset,shuffle=True, **keyword_value_dict)

def test_dataset():
    for image,label in train_dataset:
        print(image.shape,label)


def get_dataset():
    return train_dataset, validation_dataset

def get_loader():
    return train_loader, validation_loader

def validation_dataset():
    print(train_dataset.__len__(),validation_dataset.__len__())
    print("END")

def tensorflow_mnist():
    train_dataset, tf.keras.datasets.mnist.load_data()

if __name__=="__main__":
    test_dataset()