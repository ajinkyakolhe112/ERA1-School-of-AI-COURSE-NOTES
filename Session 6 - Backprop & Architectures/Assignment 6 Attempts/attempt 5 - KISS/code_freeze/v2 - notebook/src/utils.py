import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from tqdm import tqdm

def get_data(BATCH_SIZE: int):

	transforms_pipeline = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.1307,), std = (0.3081,))])

	train_dataset = torchvision.datasets.MNIST(root = './data',train = True,transform = transforms_pipeline, download = True)
	test_dataset = torchvision.datasets.MNIST(root = './data',train = False,transform = transforms_pipeline,download=True)

	train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = BATCH_SIZE,shuffle = True)
	test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = BATCH_SIZE,shuffle = True)

	return transforms_pipeline, train_dataset,test_dataset,train_loader,test_loader

def train(
	train_loader: torch.utils.data.DataLoader , 
	model: nn.Module , 
	error_func: nn.Module, 
	optimizer: torch.optim.Optimizer , 
	epoch_no: int = 0, device=None):

	# Data to plot accuracy and loss graphs
	train_losses = [None]* (epoch_no+1)
	train_accuracy = [None]*(epoch_no+1)
	train_loss_total = 0						# ! Total Training Loss Value
	correct_preds_total = 0						# ! Total Correct Preds
	train_accuracy_total = 0
	
	model.to(device)
	pbar = tqdm(train_loader)
		
	for b_id, (images, labels) in enumerate(pbar):
		
		train_loss_batch = 0		# ! Error Function
		train_accuracy_batch = 0	# ! Batch Accuracy Value
		correct_preds_batch = 0		# ! Correct Preds in Batch

		images,labels = images.to(device), labels.to(device)
		
		#Forward pass
		outputs = model(images)
		train_loss_batch = error_func(outputs, labels,reduction='mean')
		
		# Backward and optimize
		optimizer.zero_grad()
		train_loss_batch.backward()
		optimizer.step()
		
		pred_labels = outputs.argmax(dim=1, keepdim=True).squeeze()
		correct_preds_batch = torch.eq(pred_labels, labels).sum().item()
		train_accuracy_batch = 100.0 * correct_preds_batch / train_loader.batch_size
		pbar.set_description("Train: Epoch %d, Batch= %d, Batch Error Value = %f, Batch Pred Acc = %f" % (epoch_no+1,b_id+1, train_loss_batch.item(), train_accuracy_batch))
		correct_preds_total = correct_preds_total + correct_preds_batch
		train_loss_total = train_loss_batch.item()
		if (b_id + 1) % 250 == 0:
			print("Train: Epoch %d, Batch %d, Error Value = %f, Pred_Acc = %f" % (epoch_no+1,b_id+1, train_loss_batch.item(), train_accuracy_batch))
	train_losses[epoch_no] = train_loss_total / len(train_loader)
	train_accuracy[epoch_no] = 100.0 * correct_preds_total / len(train_loader)
	# print("T Epoch %d, Final Accuracy of the network %f" % (epoch_no +1 , train_accuracy[epoch_no]))

def test(
	test_loader: torch.utils.data.DataLoader , 
	model: nn.Module , 
	error_func: nn.Module, 
	epoch_no: int = 0, device=None):

	torch.set_grad_enabled(False)

	test_losses = [None]* (epoch_no + 1)
	test_accuracy = [None]* (epoch_no + 1)
	
	test_loss_total = 0
	correct_preds_total = 0
	pbar = tqdm(test_loader)

	for b_id,(images, labels) in enumerate(pbar):
		images,labels = images.to(device), labels.to(device)
		
		outputs = model(images)
		error_batch = error_func(outputs, labels, reduction='mean')
		
		pred_labels = outputs.argmax(dim=1, keepdim=True).squeeze()
		correct_preds = torch.eq(pred_labels, labels).sum().item()
		accuracy_batch = correct_preds / test_loader.batch_size * 100
		
		test_loss_total = test_loss_total + error_batch.item()
		correct_preds_total += correct_preds
		
		pbar.set_description("Test: Epoch %d, Batch= %d, Error Value = %f, Pred Acc = %f" % (epoch_no+1, b_id+1, error_batch.item(), accuracy_batch))
		if (b_id + 1) % 100 == 0:
			print("Test: Epoch %d, Batch %d, Error Value = %f, Pred_Acc = %f" % (epoch_no+1,b_id+1, error_batch.item(), accuracy_batch))
	test_losses[epoch_no] = test_loss_total / len(train_loader)
	test_accuracy[epoch_no] = 100.0 * correct_preds_total / len(test_loader)
	print("Epoch %d, Final Accuracy of the network %f" % (epoch_no +1 , 100.0 * correct_preds_total / len(test_loader)))
	torch.set_grad_enabled(True)

def test_nn_training():
	from models.model_1 import get_model
	
	BATCH_SIZE = 64
	num_classes = 10
	LR = 0.01
	NUM_EPOCHS = 10
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Data to plot accuracy and loss graphs
	train_losses = [None]*NUM_EPOCHS
	train_accuracy = [None]*NUM_EPOCHS

	test_losses = [None]*NUM_EPOCHS
	test_accuracy = [None]*NUM_EPOCHS

	transforms, train_dataset,test_dataset,train_loader,test_loader = get_data(BATCH_SIZE)
	model = get_model()
	error_func = torch.nn.functional.nll_loss
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)

	train(train_loader, model, error_func, optimizer)




if __name__ == "__main__":
	test_nn_training()