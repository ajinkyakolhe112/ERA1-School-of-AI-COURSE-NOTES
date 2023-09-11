#!/usr/bin/env python3

import torch
import torch.nn as nn

from torchinfo import summary
from torchvision import datasets,transforms
from sklearn import datasets as toysets

# in_features vs out_features

linear11 = nn.Linear(1,1)
linear12 = nn.Linear(1,2)
linear21 = nn.Linear(2,1)
linear22 = nn.Linear(2,2)
linear32 = nn.Linear(3,2)
linear23 = nn.Linear(2,3)



print("End")