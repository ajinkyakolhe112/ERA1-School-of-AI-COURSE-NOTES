"""Original file is located at: https://colab.research.google.com/drive/1gbN0fsB236mQEu3uJIpXQWx55YuRsLKV"""

import builtins # names of built-in functions and variables
dir(builtins)
# we are importing more functions & variables from libraries we will need

import torch 
#%% [markdown]
# 1. `import torch` module. It contains Tensors, Random Sampling methods group, Math Operations & torch compile . 
# 2. to check what each module is importing, execute
#   - dir(torch)
#   - vars(torch)

import torch.nn as nn
# nn basic building block for NN. Layers, Activations, Loss Functions. Each is extendible. 
# Base: nn.Containers module. Imp once are (nn.Module, nn.Sequential), nn.ParameterDict. Extra forward & backward pre & post hooks.  
# Details of nn.Module: essential methods to override init & forward. 
# Module.training<Status> represents whether this module is in training or evaluation mode.

import torch.nn.functional as F
# module for grouping of all functions in pytorch. Internally in Conv2d layer, respective function from here is called

import torch.optim as optim
# Package implementing various optimization algorithms. SGD, ADAM variations. 
# Also 3 different implementations of algorithms with different tradeoffs. 
# 3 major categories of implementations: for-loop, foreach (multi-tensor), and fused. the performance ordering of the 3 implementations is fused > foreach > for-loop
# Only Adam & AdamW have fused implementation

from torchvision import datasets, transforms 
# Seperate library, part of Pytorch. Doesn't mess up generic torch module. And specialized vision related are organized here
# 1. popular datasets, 2. model architectures, and 3. image transformations for computer vision.

#!pip install torchsummary
from torchsummary import summary
# Important for checking parameters & output size. But has been superseded by torchinfo package. 
# torchinfo has verbose customizable output compared to this one
from tqdm import tqdm


use_cuda = torch.cuda.is_available()       
device = torch.device("cuda" if use_cuda else "cpu") 
device
# cuda default accelerator for DL. Also have tpu which is accelerated for pytorch with XLA. 

# our brain has much lower clock speed, but much faster data access. So
# In Future more specialized AI accelerators, with faster memory. (currently just flops have been scaled. We have hit memory wall where fast memory hasn't been able to keep up with flops
# Whatever place parameters are stored, needs to be fast access and big storage both... Specialized hardwares of future should have at least enough memory to save entire parameters in memory 

# Default Building Blocks of Deep Learning operations are 1. Matrix Multiplication + Vector Addition(X*W + b) during forward pass & 2. Parameters Update (w = w-lr*grad). during backward 3. Gradient Calculation for a few Error Functions
# These are in billions. Any speedup in Flops or Memory of above operations will accelerate AI workload. Hence we have AI Accelerators. 
# Also because DL can improve in direct proportion of Data & Compute. That's why blindly increasing them leads to improvement.
# Computational Linear Algebra for scaling solutions to much higher dimentional values.

batch_size = 128  
# each batch leads to one update of parameters. All 128 images would be run parallely, hence speedup achieved.
# I don't know what is the upper limit of the batch_size. 

train_loader = torch.utils.data.DataLoader(
    # At the heart of PyTorch data loading utility is the torch.utils.data.DataLoader class. It represents a Python iterable over a dataset
    # torchvision.datasets for all vision datasets. Has download from internet option.
    # torchvision.datasets does tranforms of Data .. AND torch.utils.data.DataLoader does batching... 
    datasets.MNIST('../data', train=True, download=True,     
                    transform=transforms.Compose([          
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True)

#%% [markdown]
"""
# Some Notes on our naive model

We are going to write a network based on what we have learnt so far. 
The size of the input image is 28x28x1. We are going to add as many layers as required to reach RF = 32 "atleast".

Commentary
1. nn.Module Base module for Neural Network & Also Custom Layer if extending
1. Custom NN, has 1. initialization of Network layers & parameters.  2. forward pass & 3. error backward propogation
1. Base NN, would be at least when we have reached RF limit. Generally based on image, depth is max ImageSize/2. (Because current kernels is size 3)
1. Relevant formulate to calculate max depth $\Large n_{out} = \frac{(n_{in} + 2 \cdot p - k)}{s} + 1$
1. Effective Receptive Field = size of image. We need to build it in nested way with kernels.

Rearranging formulate for s = 1 , p = 1 & fixed k

$$
n_{out} = n_{in} - d \cdot (k-1)
\\
d = \frac{n_{in}}{(k-1)}
$$
Or we can just keep adding until we reach $n_{out} = 1$
"""

#%%
class FirstDNN(nn.Module):
    # nn.Module either model or layer or block of layers
    def __init__(self):
        super(FirstDNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # RECEPTIVE FIELD REACHED HERE.

        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)   
        self.conv7 = nn.Conv2d(1024, 10, 5)

    def forward(self, x): # x is the input to module or layer. x is Example or Data to be transformed by layer
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x))))) # nested function calling. Mathematically its h(g(f(x)))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x)) 
        # This has been sequential forward pass. If wanted to branch out, can do those transformations as well. 
        # And accordingly, error will be propogated automatically by autograd

        x = x.view(-1, 10) # Shape to 10 because 10 number of classes and we will be doing softmax across those values
        return F.log_softmax(x)

#%%
model = FirstDNN() # create the model
model.to(device)   # send model to device
model # Print the model

summary(model, input_size=(1, 28, 28)) # can use this output to check if calculations in model are correct or not. 

# PARAMETERS are stored in Memory
# Output Shape is also stored in memory
# Matrix operations are done on Input or Output to update Parameters

#%%
def train(model, device, train_loader, optimizer, epoch): # simple function to do entire code of training. 
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar): # main training code is this. send data batch by batch. calculate error, backpropogate and weight update acc to optmizer
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data) # output for the batch of Data
        loss = F.nll_loss(output, target) # loss for the batch of Data
        loss.backward() # loss is calculated for all backward layer
        optimizer.step() # Here, optimizer updates weight according to contribution in that Error Value.. 
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


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

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#%%
device = "cpu"
device = torch.device(device)

model = FirstDNN()
model.to(device)   # send model to device

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)