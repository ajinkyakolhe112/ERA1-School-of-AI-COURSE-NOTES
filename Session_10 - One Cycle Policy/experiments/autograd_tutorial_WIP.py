#%%
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
#%%
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

model = resnet18(weights=ResNet18_Weights.DEFAULT)

#%%
prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()

#%%
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step()

#%%
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3*torch.pow(a,3) - torch.pow(b,2)
# %%
