#%%
from torchinfo import summary
import torch

#%%
model = torch.nn.Sequential()
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W)
# where H and W are expected to be at least 224
model = torch.hub.load("pytorch/vision:v0.10.0", model = "alexnet",pretrained=True)

model(torch.randn(3,224,224))
summary(model,input_size=(3,256,256),verbose=2,
	col_names=["input_size","kernel_size", "output_size", "num_params", "params_percent"],col_width=20);
# %%
