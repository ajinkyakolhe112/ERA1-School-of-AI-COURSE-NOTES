import torch
import numpy as np
import tensorflow as tf

#%% LIBRARIES, FUNCTIONS & MEMORY
"Memory: However, in general, you can expect the NumPy, Torch, and TensorFlow modules to take up around 60 MB, 100 MB, and 150 MB of memory"
"Functions and how they are named in each Library"

"base functions in numpy"
"base functions in torch"
"base functions in tensorflow"
len(dir(torch)),len(dir(tf)),dir(tf.keras),len(dir(np))  # (1251, 342, 35, 600)

"In torch, mostly functions are limited & directly used. Low Level, limited number. Less files & modules implemented. No need to organize much"
"Tensorflow, low & high level. So module name design is importantly"
"np is also for number array & tensor is also for numerical tensor in Pytorch. np is specialized for numerical processing. Pytorch is specialized for DNN implemented by Facebook" "np is wider in scope than Pytorch"

"NumPy(1300+) has more functions implemented than PyTorch (600+). \
NumPy is a general-purpose scientific computing library, while PyTorch is a deep learning library."


#%% TENSOR DIMENTION
"base functions of torch.Tensor"
len(dir(torch.Tensor(1))) # 722

"ndim, dim(), torch.linalg.matrix_rank or len(Tensor.shape)"

testTensor = torch.as_tensor(1)
t0d = torch.as_tensor(1)
t1d = torch.randn(1) # In one dimention. Infinite growth in as many columns in single example. 
t2d = torch.randn(1,1)
t3d = torch.randn(1,1,1)

ANY_NUMBER = 100
t1d = torch.randn(ANY_NUMBER) # torch.randn(1).ndim == torch.randn(100).ndim torch.randn(10000)

t2d = torch.randn(1,1)
t2d = torch.randn(104,545)
t2d = torch.randn(5,5)

t3d = torch.randn(1,1,1)
t3d = torch.randn(10,10,10)

t4d = torch.randn(0,1,2,3)
t5d = torch.randn(0,1,2,3,4)
"Everything is called tensor, but ideally, it should be called n-dimentional tensor"
"Regression is 1d Tensor, Images are 3d Tensors, and for adding more dimentions, we need to go 4d, 5d etc"

#%% TENSOR CREATION FUNCTIONS
"rand, randn, randint"
"torchvision.datasets"

#%% TENSOR ESSENTIAL PROPERTIES
"device, requires_grad, dtype"

#%% TENSOR IMP METHODS
"shape, reshape, squeeze, unsqueeze, flatten, view"
"maths functions in Tensor. abs, sqr, log, softmax ...etc "

#%% TENSOR GRAPH & GRADIENT & AUTOGRAD

tensor1 = torch.randn(1,requires_grad=True)
tensor2 = torch.randn(1,requires_grad=True)
tensorSqr = tensor1 * tensor1
tensorMult = tensor1 * tensor2 + tensorSqr
tensorAdd = tensor1 + tensor2
tensorMult.backward()

X = torch.randn(3, requires_grad=True)
W = torch.randn(3, requires_grad=True)
b = torch.randn(1,requires_grad=True)
z = X * W + b
a = torch.exp(z)

x = torch.randn(1,requires_grad=True)
square = x**2
cube = x**3
four = x**4
final = four + cube + square
final.backward()

#%% TENSOR GRAPH OF OPERATIONS


print("END")