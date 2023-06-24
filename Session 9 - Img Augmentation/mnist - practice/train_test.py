import torch
import torch.nn as nn
import tqdm.tqdm

def train_model(train_loader, model, error_func, optimizer, device=None, epoch_no=1):
	# Follow kitchen philosophy. Make everything ready for training loop
	pbar = tqdm(train_loader)
	model.to(device)
	optimizer.zero_grad()

	for batch_no, (image_tensors, labels) in enumerate(pbar):
		x_actual, y_actual = image_tensors.to(device),labels.to(device)

		# Training Loop
		y_pred = model(x_actual)
		loss = error_func(y_pred,y_actual, reduction='mean')
		loss.backward()
		optimizer.step()
		optimizer.zero_grad() # Needed inside forloop. Also keep optimizer outside too. Pythonic

		# Metrics
		pbar.set_description(f'Batch Train Loss= {loss.item():0.4f}')
	pass

def test_model(test_loader, model, error_func, device=None, epoch_no=1):
	pass

if __name__ == "__main__":
	from dataset_load import *
	from model_dev import *

	model = Baseline()
	error_func = nn.functional.nll_loss
	optimizer_name = torch.optim.SGD
	optimizer = optimizer_name(params = model.parameters(), lr = 0.01)

	train_model(train_loader, model, error_func, optimizer)
	pass