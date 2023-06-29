import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

def train_model(train_loader, model, error_func, optimizer, device=None, epoch_no=1):
	# Follow kitchen philosophy. Make everything ready for training loop
	pbar = tqdm(train_loader)
	model.train(mode=True)
	model.to(device)
	optimizer.zero_grad()

	#Training Loop Metrics
	total_correct = 0
	training_metrics = {
			"batch_loss": [],
			"batch_acc": [],
			"acc_total": [],
			
			# "batch_neurons_corrected": [],
			# "batch_gradients": [],
			# "customizable": [],
			"batch_correct": [],
			"batch_no": [],
		}

	for batch_no, (image_tensors, labels) in enumerate(pbar):
		x_actual, y_actual = image_tensors.to(device),labels.to(device)

		# Training Loop
		y_pred = model(x_actual)
		loss = error_func(y_pred,y_actual, reduction='mean')
		loss.backward()
		optimizer.step()
		optimizer.zero_grad() # Needed inside forloop. Also keep optimizer outside too. Pythonic

		# Training Loop Metrics
		loss_batch = loss.item()
		y_pred,y_actual
		y_pred_class = y_pred.argmax(dim=1,keepdim=False)
		batch_correct_pred = y_pred_class.eq(y_actual).count_nonzero().item()
		total_correct = total_correct + batch_correct_pred
		total_processed = (batch_no + 1) * train_loader.batch_size
		total_acc = (100.0 * total_correct) / total_processed

		pbar.set_description(f'Batch No= {batch_no}, Batch Loss= {loss_batch:0.4f}, Batch Correct= {batch_correct_pred:3d}, Acc Total = {total_acc:0.4f}')
		
		training_metrics["batch_loss"].append(loss.item())
		training_metrics["batch_acc"].append( 1.0 * batch_correct_pred / train_loader.batch_size )
		training_metrics["acc_total"].append(total_acc)
		training_metrics["batch_correct"].append(batch_correct_pred)
		training_metrics["batch_no"].append(batch_no+1)
		wandb.log({
		"batch_no": batch_no,
		"acc": training_metrics["acc_total"][-1], 
		"loss": training_metrics["batch_loss"][-1], 
		"batch_acc": training_metrics["batch_acc"][-1],
		"batch_correct": training_metrics["batch_correct"][-1],
		})


		# training_metrics["customizable"].append("")
	return training_metrics

def test_model(test_loader, model, error_func, device=None, epoch_no=1):
	pbar = tqdm(test_loader)

	for batch_no,(image_tensors, labels) in enumerate(pbar):
		x_actual, y_actual = image_tensors.to(device),labels.to(device)
		
		y_pred = model(x_actual)
		loss = error_func(y_pred,y_actual, reduction='mean')
		
		# Test Loop Metrics
		
	pass

def wandb_init():
	wandb.init(
		name= "configuring wandb",
		project="mnist",
		config = {
			"dataset": "MNIST",
			"learning_rate": 0.01,
		}
	)



if __name__ == "__main__":
	from dataset_load import *
	from model_dev import *

	model_s9 = S9_Baseline()
	error_func = nn.functional.nll_loss
	optimizer = torch.optim.SGD(params = model_s9.parameters(), lr = 0.01)
	wandb_init()

	training_metrics = train_model(train_loader, 	model_s9, error_func, optimizer)


	test_model(	test_loader, 	model_s9, error_func)
	pass