from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as torch_optimizer

# Data to plot accuracy and loss graphs
training_losses_epochwise = []
training_accuracy_epochwise = []

test_losses_epochwise = []
test_accuracy_epochwise = []

def train(train_loader, model, errorFun, optimizer, epoch_no, device=None):
	model.train()
	model.to(device)
	pbar = tqdm(train_loader)
	
	error_value_total = 0
	correct_preds_total = 0
	total_processed = 0

	correct_preds_batch = 0

	"Training Loop"
	for batch_idx, (x_batch, y_actual) in enumerate(pbar):
		x_batch, y_actual = x_batch.to(device), y_actual.to(device)
		
		"1. Forward Pass. Builds graph"
		y_pred_probs = model(x_batch)
		
		"2. Error Value for model"
		error_value_batch = errorFun(y_pred_probs, y_actual, reduction = "mean")
		
		"3. Error wrt W, backprogated to each W"
		error_value_batch.backward()
		
		"4. Update parameters W in direction of Error_gradient"
		optimizer.step()
		
		"5. Clean graph"
		optimizer.zero_grad()

		"Acc Value for Humans"
		y_pred_class = y_pred_probs.argmax(dim=1)
		comparison = torch.eq(y_pred_class, y_actual)
		correct_preds_batch = comparison.sum().item()
		accuracy_batch = 100.0 * (correct_preds_batch / x_batch.shape[0])
		
		pbar.set_description("Batch= %d, Batch Error = %f, Batch Accuracy = %f" %
			(batch_idx, error_value_batch.item(),accuracy_batch))
		
		error_value_total += error_value_batch
		correct_preds_total += correct_preds_batch
		total_processed = total_processed + x_batch.shape[0]


	
	training_error_value_avg = error_value_total/total_processed
	training_accuracy_total = 100 * (total_processed/total_processed)
	
	training_losses_epochwise.append(training_error_value_avg)
	training_accuracy_epochwise.append(training_accuracy_total)

def test(test_dataloader, model, errorFun, device=None):
	model.eval()
	model.to(device)
	pbar = tqdm(test_dataloader)
	
	test_loss_total = 0
	correct_preds_total = 0
	processed_total = 0

	with torch.no_grad():
		test_loss_batch = 0
		test_correct_preds_batch = 0
		test_correct_preds_total = 0
		
		for data, target in pbar:
			data, target = data.to(device), target.to(device)
			
			"1. Calculate Inference"
			output = model(data)
			"2. Calcualate Error"
			loss_batch = errorFun(output, target, reduction='sum').item()   # sum up batch loss
			
			y_pred_class = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
			comparison = torch.eq(y_pred_class, target)
			# test_correct_preds_batch = target[comparison]
			correct_pred_indexes = torch.where(comparison == True)
			accuracy_batch = 100.0 * ( len(correct_pred_indexes) / data.shape[0])
			
			test_loss_total += loss_batch
			test_correct_preds_total += test_correct_preds_batch

			processed_total += data.shape[0]
			pbar.set_description("Batch: error = %f \t, accuracy %f" % (loss_batch,accuracy_batch))
	
	test_loss_avg = test_loss_total / processed_total
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
		format(test_loss_avg, correct, processed_total,100. * correct / processed_total))

if __name__ =="__main__":
	from data import *
	from models import *
	# Test
	train_dataloader, test_dataloader
	model = FirstDNN()
	errorFun = torch.nn.functional.nll_loss

	optimizer = torch_optimizer.SGD(model.parameters(), lr=0.01, momentum=0.9)
	device = torch.device("mps")

	for epoch in range(1, 2):
		train(train_dataloader, model, errorFun, optimizer, epoch)
		test(test_dataloader, model, errorFun)
	
	

		