"""Linear Regression + Linear Algebra.ipynb
Original file is located at
    https://colab.research.google.com/drive/1KSC9vBmwslYyi67l305-aOcHrj4G8R4M
"""

import torch
import torch.nn as nn
from torchinfo import summary

import torch # Tensor Library
import numpy as np # ndim Array Library
from sympy import Matrix
from IPython.display import display,Latex,Math

# input data = single line of n dimentional columns. (vector)
linear23 = nn.Linear(2,3) # Output: 3 Neurons, Input: 2 Column 1D Example , Parameters: Neurons * Input Vector

# Commented out IPython magic to ensure Python compatibility.
# %%latex
# 
# X = \begin{bmatrix} x_{1} ,\\x_{2} \end{bmatrix} ,
# W^0 = \begin{bmatrix} w_{1} ,& w_{2} \end{bmatrix}\\
# W = \begin{bmatrix} neuron^0 ,\\ neuron^1, \\ neuron^2 \end{bmatrix}
# \\
# W = \begin{bmatrix}
# \begin{bmatrix} w^0_{0} ,& w^0_{1}\end{bmatrix}\\ 
# \begin{bmatrix} w^1_{0} ,& w^1_{1} \end{bmatrix}\\
# \begin{bmatrix} w^2_{0} ,& w^2_{1}\end{bmatrix}\\
# \end{bmatrix}
# 
# \\
# X_{transformed} = X \odot W + b

linear23.weight.shape
inputX = torch.rand(2)
inputX.shape

weight = linear23.weight

linear23,linear23.weight, linear23.bias

vars(linear23) # nn.Linear -> nn.Module

linear23(torch.rand(2,1))

X = torch.rand(100,3)
y = torch.rand(100,1)

regression = nn.Sequential(
    nn.Linear(3,5),
    nn.ReLU(),
    nn.Linear(5,1),
    # nn.ReLU(),
    # nn.Linear(2,3),
    # nn.ReLU(),
    # nn.Linear(3,1)
)

yPredicted = regression(X)
lossFun = nn.MSELoss()
lossValue = lossFun(yPredicted,y)
lossValue.backward()
lr = 0.001
for params in regression.parameters():
  params = params - lr * params.grad

vars(regression)

for i in regression.state_dict():
  print(i)

regression

dir(regression)

type(regression[0].weight)

from torchviz import make_dot
make_dot(yPredicted.mean(),params=dict(regression.named_parameters()))

make_dot(yPredicted.mean(),params=dict(regression.named_parameters()),show_attrs=True,show_saved=True)

lossValue.backward()

vars(regression[-1])

lossValue.grad_fn

regression[-1].weight.grad

lossValue.backward()

regression[-1].weight.grad

regression.named_parameters()

dir(regression[-1].weight)

lossValue.backward()
vars(regression[-1])
regression[-1].grad

yDummy = torch.rand(1)

loss(yPredicted,loss)

vars(regression[0])

regression[0]

inputX = torch.rand(1,5,5)
test = nn.Sequential(
    nn.Conv2d(1,1,3),
    nn.Flatten(),
    nn.Linear(9,10), # Error of This. Flatten solves the problem
    nn.LogSoftmax()
)

yPredicted = test(inputX)
#nn.Conv2d(1,1,3)(1,5,5)

lossFn = nn.NLLLoss()

yPredicted

nn.functional.softmax(yPredicted)

lossFn(nn.functional.log_softmax(torch.rand(1,10)),nn.functional.log_softmax(torch.rand(1,10)))

torch.nn.functional.softmax(torch.rand(1,10))

test(inputX).shape

torch.rand(2) # single vector = single ROW with n dimentions of column
torch.rand(2)  # vector of 2 column dimentions. 
torch.rand(3)  # vector of 3 column dimentions
torch.rand(10) # vector of 10 column dimentions

linear23(torch.rand(2))

linear23(torch.rand(3))

# Commented out IPython magic to ensure Python compatibility.
# %%latex
# 
# X = \begin{bmatrix} x_{1} ,\\x_{2}, \\x_{3},\\ \vdots\\ x_{in}\end{bmatrix} ,
# W = \begin{bmatrix} w_{1} ,\\w_{2}, \\w_{3},\\ \vdots\\ w_{in}\end{bmatrix}

# Commented out IPython magic to ensure Python compatibility.
# %precision 2
display(tmp)

# Commented out IPython magic to ensure Python compatibility.
# %precision 2

"""
Question. 
Why do you think, we still have high level library like keras and also low level library like pytorch?

"""

# Linear Regression with keras Dense Layer. - Weight & Bias done in keras
# Linear Regression with nn.Linear Layer - weight & bias done automatically
# Linear Regression with custom nn.Linear Layer - weights & bias as parameters