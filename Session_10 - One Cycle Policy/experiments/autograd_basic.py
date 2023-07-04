#%%
from torch.autograd import grad
x1 = torch.randn(1, requires_grad=True)
x2 = torch.randn(1, requires_grad=True)
w1 = torch.randn(1, requires_grad=False)
w2 = torch.randn(1, requires_grad=False)

output = nn.functional.relu(x1 * w1 + x2 * w2 )

d_output_d_x = grad(inputs = [x1,x2], outputs = output)
d_output_d_x

#%%
output = nn.functional.relu(x1 * w1 + x2 * w2 )
opt = torch.optim.SGD(params = [x1, x2], lr = 0.01)
output.backward()

x1.grad,x2.grad