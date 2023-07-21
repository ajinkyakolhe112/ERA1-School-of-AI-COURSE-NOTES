import torch
from loguru import logger

cifar = torch.ones(1,3,5,5)

l1 = torch.nn.Conv2d(3,4,(3,3)) # input size = 5*5*3 . output size = 3*3*4
l2 = torch.nn.Conv2d(4,4,(3,3),padding="same") # input size = 3*3*4 , output size = 3*3*4

l1(cifar).shape
l2(l1(cifar)).shape

l1_output = l1(cifar)
l2_output = l2(l1_output)

l3_output = l1_output + l2_output
l3_output = torch.add(l1_output, l2_output)
l3_output.shape
logger.debug(f'{l1_output}')
logger.debug(f'{l2_output}')
logger.debug(f'{l3_output}')
logger.debug(f'{l1_output}+{l2_output}={l3_output}')

nn.ModuleDict({
    "conv1": nn.Conv2d(),
    "conv2": nn.Conv2d(),
})

logger.debug(f'')
