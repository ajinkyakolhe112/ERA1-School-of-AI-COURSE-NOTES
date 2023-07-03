#%%
import torch
import torch.nn as nn
from torchinfo import summary

import torch # Tensor Library
import numpy as np # ndim Array Library
from sympy import Matrix
from IPython.display import display,Latex,Math

# input data = single line of n dimentional columns. (vector)
linear23 = nn.Linear(2,3) # Output: 3 Neurons, Input: 2 Column 1D Example , Parameters: Neurons * Input Vector

#%% [markdown]
"""
$$
X_{batch} = \begin{bmatrix} x_{1} ,\\x_{2} \end{bmatrix} ,\\
neuron^{0} = \ W^{0} = \begin{bmatrix} w_{1} ,& w_{2} \end{bmatrix}\\

W = \begin{bmatrix} neuron^{0} ,\\[0.1cm] neuron^{1}, \\[0.1cm] neuron^{2} \end{bmatrix}\\
$$
"""
#%% [markdown]
"""
$$
\fbox{45}\\
a_{[3][4]}
$$
"""
#%% [markdown]
"""
$$
\begin{align*}
W = \begin{bmatrix}
\begin{bmatrix} w^0_{0}\ ,& w^0_{1}\end{bmatrix}\ \\
\begin{bmatrix} w^1_{0}\ ,& w^1_{1} \end{bmatrix}\ \\
\begin{bmatrix} w^2_{0}\ ,& w^2_{1}\end{bmatrix}\ \\
\end{bmatrix}
\\
X_{transformed} = X \odot W + b\\
\end{align*}
$$
"""

#%%
linear23.weight.shape
inputX = torch.rand(2)
inputX.shape

weight = linear23.weight

linear23,linear23.weight, linear23.bias

vars(linear23) # nn.Linear -> nn.Module

linear23(torch.rand(2,1))

X = torch.rand(100,3)
y = torch.rand(100,1)

regression_model = nn.Sequential(
    nn.Linear(3,5),
    nn.ReLU(),
    nn.Linear(5,1),
    # nn.ReLU(),
    # nn.Linear(2,3),
    # nn.ReLU(),
    # nn.Linear(3,1)
)
#%% [markdown]
"""
$$
Y_{pred} = regression\_model(X_{actual})\\
error\_value\ =\ error\_func\ (Y_{pred},Y_{actual})\\
error\_value.backward() = (\sum(X_{batch},W) - Y).backward()\\
optimizer.step() \\
optimizer.zero\_grad() \\
$$
"""
#%%

y_predicted = regression_model(X)
error_func = nn.MSELoss()
error_value = error_func(y_predicted,y)
error_value.backward()

loss_value = error_value
#%%
lr = 0.001
for params in regression_model.parameters():
  params = params - lr * params.grad

#%%
vars(regression_model)

for i in regression_model.state_dict():
  print(i)

regression_model

dir(regression_model)

type(regression_model[0].weight)
#%%
from torchviz import make_dot
make_dot(y_predicted.mean(),params=dict(regression_model.named_parameters()))

make_dot(y_predicted.mean(),params=dict(regression_model.named_parameters()),show_attrs=True,show_saved=True)

#%%
loss_value.backward()

vars(regression_model[-1])

loss_value.grad_fn

regression_model[-1].weight.grad

loss_value.backward()

regression_model[-1].weight.grad

regression_model.named_parameters()

dir(regression_model[-1].weight)

loss_value.backward()
vars(regression_model[-1])
regression_model[-1].grad
#%%
y_dummy = torch.rand(1)

loss(y_predicted,loss)

vars(regression_model[0])

regression_model[0]

inputX = torch.rand(1,5,5)
test = nn.Sequential(
    nn.Conv2d(1,1,3),
    nn.Flatten(),
    nn.Linear(9,10), # Error of This. Flatten solves the problem
    nn.LogSoftmax()
)

y_predicted = test(inputX)
#nn.Conv2d(1,1,3)(1,5,5)

error_func = nn.NLLLoss()
error_value = error_func(y_predicted,y_actual)
error_value = nn.functional.log_softmax(y_predicted,y_actual)

#%%

torch.nn.functional.softmax(torch.rand(1,10))

test(inputX).shape

torch.rand(2) # single vector = single ROW with n dimentions of column
torch.rand(2)  # vector of 2 column dimentions. 
torch.rand(3)  # vector of 3 column dimentions
torch.rand(10) # vector of 10 column dimentions

linear23(torch.rand(2))

linear23(torch.rand(3))

#%% [markdown]
"""
$$
X = \begin{bmatrix} x_{1} ,\\x_{2}, \\x_{3},\\ \vdots\\ x_{in}\end{bmatrix} ,
W = \begin{bmatrix} w_{1} ,\\w_{2}, \\w_{3},\\ \vdots\\ w_{in}\end{bmatrix}

Commented out IPython magic to ensure Python compatibility.
%precision 2
display(tmp)

Commented out IPython magic to ensure Python compatibility.
%precision 2
$$
"""

# Question. 
# Why do you think, we still have high level library like keras and also low level library like pytorch?
# Linear Regression with keras Dense Layer. - Weight & Bias done in keras
# Linear Regression with nn.Linear Layer - weight & bias done automatically
# Linear Regression with custom nn.Linear Layer - weights & bias as parameters
# %%
