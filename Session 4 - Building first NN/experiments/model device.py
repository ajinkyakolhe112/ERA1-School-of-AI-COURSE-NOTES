import torch
#%%
if torch.cuda.is_available():
	accelerator = "cuda"
elif torch.backends.mps.is_available():
	if not torch.backends.mps.is_built():
		accelerator = "mps"
	else:
		accelerator = "cpu"
else:
	accelerator = "cpu"

if torch.backends.cuda.is_available():
	accelerator = "cuda"
elif torch.backends.mps.is_available():
	if not torch.backends.mps.is_built():
		accelerator = "mps"
	else:
		accelerator = "cpu"
else:
	accelerator = "cpu"

device = torch.device(accelerator)
device
