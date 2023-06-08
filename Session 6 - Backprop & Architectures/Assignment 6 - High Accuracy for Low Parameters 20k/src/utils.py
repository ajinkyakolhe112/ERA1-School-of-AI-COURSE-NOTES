from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as torch_optimizer

# Data to plot accuracy and loss graphs
training_losses_epochwise = []
training_accuracy_epochwise = []

test_losses_epochwise = []
test_accuracy_epochwise = []

"1. X,Y -> 2. Y_Pred = Model(X,W) -> Output -> 3. Error = Output - Observed -> 4. Error Backward -> 5. Parameter Update"
def train(train_dataloader, model, errorFun, optimizer, epoch_no, device=None):
	model.train()
	model.to(device)
	pbar = tqdm(train_dataloader)
	
	train_loss_total = 0
	correct_preds_total = 0
	processed_total = 0

	"Training Loop"
	for batch_idx, (x_batch, y_actual) in enumerate(pbar):
		"1"
		x_batch, y_actual = x_batch.to(device), y_actual.to(device)
		
		"2. Forward Pass. Builds graph"
		y_pred_probs = model(x_batch)
		
		"3. Error Value for model"
		error_value_batch = errorFun(y_pred_probs, y_actual, reduction = "mean")
		
		"4. Error wrt W, backprogated to each W. dE/dW"
		error_value_batch.backward()
		
		"5. Update parameters W in direction of Error_gradient"
		"W = W - dE/dW* learning_rate"
		for parameter in model.parameters():
			parameter = parameter - parameter.grad * optimizer.defaults['lr']
		# optimizer.step()
		
		"6. Clean graph"
		optimizer.zero_grad()

		"Acc Value for Humans"
		y_pred_class = y_pred_probs.argmax(dim=1,keepdim=False)
		comparison = torch.eq(y_pred_class, y_actual)
		correct_preds_batch = comparison.sum().item()
		correct_pred_indexes = torch.where(comparison == True)
		correct_preds_batch = y_actual[correct_pred_indexes].numel()
		
		accuracy_batch = 100.0 * (correct_preds_batch / x_batch.shape[0])
		
		pbar.set_description("TRAIN: Batch= %d, Batch Error = %f, Batch Accuracy = %f" %
			(batch_idx, error_value_batch.item(),accuracy_batch))
		
		train_loss_total += error_value_batch.item()
		correct_preds_total += correct_preds_batch
		processed_total = processed_total + x_batch.shape[0]


	training_error_value_avg = train_loss_total /processed_total
	training_accuracy_total = 100 * (processed_total/processed_total)
	
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
		for data, target in pbar:
			data, target = data.to(device), target.to(device)
			
			"1. Calculate Inference"
			output = model(data)
			"2. Calcualate Error"
			error_value_batch = errorFun(output, target, reduction='sum').item()   # sum up batch loss
			
			y_pred_class = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
			comparison = torch.eq(y_pred_class, target)
			# test_correct_preds_batch = target[comparison]
			correct_pred_indexes = torch.where(comparison == True)
			correct_preds_batch = target[correct_pred_indexes].numel()
			accuracy_batch = 100.0 * ( correct_preds_batch / data.shape[0])
			
			test_loss_total += error_value_batch
			test_correct_preds_total += correct_preds_batch

			processed_total += data.shape[0]
			pbar.set_description("TEST: Batch: error = %f \t, accuracy %f" % (error_value_batch,accuracy_batch))
	
	test_loss_avg = test_loss_total / processed_total
	print('\nTest set: Average loss total: %f, Accuracy total: %f" \n'.format(test_loss_avg,100.0 * test_correct_preds_total / processed_total))

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
	
	

		