#%%
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#%%
# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

""" # Dataset and Creating Train/Test Split """

train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

#%%
""" # Dataloader Arguments & Test/Train Dataloaders """

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

#%%
"""# Data Statistics: It is important to know your data very well. Let's check some of the statistics around our data and how it actually looks like """

# We'd need to convert it into Numpy! Remember above we have converted it into tensors already

train_data = train.train_data
train_data = train.transform(train_data.numpy())

print('[Train]')
print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
print(' - Tensor Shape:', train.train_data.size())
print(' - min:', torch.min(train_data))
print(' - max:', torch.max(train_data))
print(' - mean:', torch.mean(train_data))
print(' - std:', torch.std(train_data))
print(' - var:', torch.var(train_data))

dataiter = iter(train_loader)
images, labels = next(dataiter)

print(images.shape)
print(labels.shape)

#%%
# Let's visualize some of the images
# %matplotlib inline
import matplotlib.pyplot as plt

plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

""" ## MORE: It is important that we view as many images as possible. This is required to get some idea on image augmentation later on """

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')

#%%
"""# How did we get those mean and std values which we used above?

Let's run a small experiment """

# simple transform
simple_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                      #  transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values.
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])
exp = datasets.MNIST('./data', train=True, download=True, transform=simple_transforms)
exp_data = exp.train_data
exp_data = exp.transform(exp_data.numpy())

print('[Train]')
print(' - Numpy Shape:', exp.train_data.cpu().numpy().shape)
print(' - Tensor Shape:', exp.train_data.size())
print(' - min:', torch.min(exp_data))
print(' - max:', torch.max(exp_data))
print(' - mean:', torch.mean(exp_data))
print(' - std:', torch.std(exp_data))
print(' - var:', torch.var(exp_data))

#%%
"""# The model
Let's start with the model we first saw
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # 28 > 28 | 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)    # 28 > 28 |  5
        self.pool1 = nn.MaxPool2d(2, 2)                 # 28 > 14 | 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)   # 14> 14 | 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  #14 > 14 | 14
        self.pool2 = nn.MaxPool2d(2, 2)                 # 14 > 7 | 28
        self.conv5 = nn.Conv2d(256, 512, 3)             # 7 > 5 | 30
        self.conv6 = nn.Conv2d(512, 1024, 3)            # 5 > 3 | 32 | Output: 3*3*1024 | Kernel: (3*3*512)*(1024)
        self.conv7 = nn.Conv2d(1024, 10, 3)             # 3 > 1 | 34 | Output: 1*1*10   | Kernel: (3*3*1024)*(10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        # x = F.relu(self.conv7(x))
        x = self.conv7(x)
        x = x.view(-1, 10)                              # 1*1*10 > (-1,10)
        return F.log_softmax(x, dim= 1) # Two dimentions. dim 0 & dim 1
        return F.log_softmax(x, dim=-1) # -1 means last dimention which is dim 1 in this case

#%%
"""# Model Params
Can't emphasize on how important viewing Model Summary is.
Unfortunately, there is no in-built model visualizer, so we have to take external help
"""

# !pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(1, 28, 28))

#%%
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 28, 28]             320
            Conv2d-2           [-1, 64, 28, 28]          18,496
         MaxPool2d-3           [-1, 64, 14, 14]               0
            Conv2d-4          [-1, 128, 14, 14]          73,856
            Conv2d-5          [-1, 256, 14, 14]         295,168
         MaxPool2d-6            [-1, 256, 7, 7]               0
            Conv2d-7            [-1, 512, 5, 5]       1,180,160
            Conv2d-8           [-1, 1024, 3, 3]       4,719,616
            Conv2d-9             [-1, 10, 1, 1]          92,170
================================================================
Total params: 6,379,786
Trainable params: 6,379,786
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.51
Params size (MB): 24.34
Estimated Total Size (MB): 25.85
----------------------------------------------------------------
"""
#%%
from torchinfo import summary as summary2
summary2(model, input_size=(1,28,28), verbose=2, col_names=["kernel_size", "num_params", "params_percent", "mult_adds" ], col_width=15)

#%%
"""
====================================================================================================
Layer (type:depth-idx)                   Kernel Shape    Param #         Param %         Mult-Adds
====================================================================================================
Net                                      --              --                   --         --
├─Conv2d: 1-1                            [3, 3]          320               0.01%         286,720
│    └─weight                            [1, 32, 3, 3]   ├─288
│    └─bias                              [32]            └─32
├─Conv2d: 1-2                            [3, 3]          18,496            0.29%         33,144,832
│    └─weight                            [32, 64, 3, 3]  ├─18,432
│    └─bias                              [64]            └─64
├─MaxPool2d: 1-3                         2               --                   --         --
├─Conv2d: 1-4                            [3, 3]          73,856            1.16%         132,349,952
│    └─weight                            [64, 128, 3, 3] ├─73,728
│    └─bias                              [128]           └─128
├─Conv2d: 1-5                            [3, 3]          295,168           4.63%         1,057,882,112
│    └─weight                            [128, 256, 3, 3] ├─294,912
│    └─bias                              [256]           └─256
├─MaxPool2d: 1-6                         2               --                   --         --
├─Conv2d: 1-7                            [3, 3]          1,180,160        18.50%         3,021,209,600
│    └─weight                            [256, 512, 3, 3] ├─1,179,648
│    └─bias                              [512]           └─512
├─Conv2d: 1-8                            [3, 3]          4,719,616        73.98%         14,498,660,352
│    └─weight                            [512, 1024, 3, 3] ├─4,718,592
│    └─bias                              [1024]          └─1,024
├─Conv2d: 1-9                            [3, 3]          92,170            1.44%         921,700
│    └─weight                            [1024, 10, 3, 3] ├─92,160
│    └─bias                              [10]            └─10
====================================================================================================
Total params: 6,379,786
Trainable params: 6,379,786
Non-trainable params: 0
Total mult-adds (G): 18.74
====================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 1.38
Params size (MB): 25.52
Estimated Total Size (MB): 26.90
====================================================================================================
"""

#%%

"""# Training and Testing

All right, so we have 6.3M params, and that's too many, we know that. But the purpose of this notebook is to set things right for our future experiments.

Looking at logs can be boring, so we'll introduce **tqdm** progressbar to get cooler logs.

Let's write train and test functions
"""

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

#%%
"""# Let's Train and test our model"""

model =  Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 3
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

#%%
train_losses

t = [t_items.item() for t_items in train_losses]

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(t)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

