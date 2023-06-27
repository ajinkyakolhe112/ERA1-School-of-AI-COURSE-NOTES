import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

class Metrics:
	def __init__(s):
		s.train_loss_batches = []
		s.train_acc_batches = []
		
		s.test_loss_batches = []
		s.test_acc_batches = []
		
		s.train_loss_epochs = []
		s.train_acc_epochs = []
		
		s.test_loss_epochs = []
		s.test_acc_epochs = []

metrics = Metrics()

def train_model(train_loader, model, error_func, optimizer, device=None):
	pbar = tqdm(train_loader)
	model.train(mode=True)
	model.to(device)
	
	total_correct = 0
	for batch_no,(images_batch,labels_batch) in enumerate(pbar):
		images_batch.to(device),labels_batch.to(device)
		
		y_pred = model(images_batch)
		loss = error_func(y_pred, labels_batch, reduction="mean")
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		
		loss_batch = loss.item()
		target_preds = y_pred.argmax(dim=1,keepdim=False)
		correct_preds = torch.eq(target_preds,labels_batch).count_nonzero().item()
		total_correct = total_correct + correct_preds
		total_acc = 100 * float(total_correct) / ((batch_no+1)*train_loader.batch_size)
		
		metrics.train_loss_batches.append(loss_batch)		# Because inside forloop. It's printing train_loss per batch
		metrics.train_acc_batches.append(total_acc)			# When outside loop, it will print train_loss per epoch
		
		message = f"Batch_id= {batch_no}, Batch Loss: {loss_batch:0.4f}, Correct: {correct_preds:2d}, Total Acc {total_acc:0.4f}"
		pbar.set_description(message)


def test_model(test_loader, model, error_func, device=None):
	model.eval()
	# DOESN"T HAVE MODE PARAMETER LIKE model.train()
	# ERROR: model.eval(True)
	# ERROR: model.eval(mode=True)
	model.to(device)
	
	test_loss_total = 0
	test_correct_total = 0
	with torch.no_grad():
		for batch_no,(images,labels) in enumerate(tqdm(test_loader)):
			images.to(device),labels.to(device)
			
			target_pred = model(images)
			# ERROR: loss = error_func(target,target_pred, reduction="sum")
			loss = error_func(target_pred, labels, reduction="sum")
			
			test_loss_total = test_loss_total + loss.item()
			correct_preds = torch.eq(target_pred.argmax(dim=1,keepdim=False),labels).count_nonzero().item()
			test_correct_total = test_correct_total + correct_preds
			test_total_acc = 100 * float(test_correct_total) / ((batch_no+1)*test_loader.batch_size)
			
			metrics.test_loss_batches.append(loss.item()/test_loader.batch_size)
			metrics.test_acc_batches.append(test_total_acc)
			
		test_loss_avg = test_loss_total / len(test_loader.dataset)
		test_acc_total = 100 * float(test_correct_total) / len(test_loader.dataset)
		metrics.test_loss_epochs.append(test_loss_avg)
		metrics.test_acc_epochs.append(test_acc_total)


def plot_train_test_loss(train_loss, train_acc, test_loss, test_acc):
	fig, axs = plt.subplots(2,2,figsize=(15,10))
	axs[0, 0].plot(train_loss)
	axs[0, 0].set_title("Training Loss")
	axs[0][0].set_xlabel("batches")
	axs[0][0].set_ylabel("loss")

	axs[1, 0].plot(train_acc)
	axs[1, 0].set_title("Training Accuracy")
	axs[1, 0].set_xlabel("batches")
	axs[1, 0].set_ylabel("acc %")

	axs[0, 1].plot(test_loss)
	axs[0, 1].set_title("Test Loss")
	axs[0, 1].set_xlabel("batches")
	axs[0, 1].set_ylabel("loss")


	axs[1, 1].plot(test_acc)
	axs[1, 1].set_title("Test Accuracy")
	axs[1, 1].set_xlabel("batches")
	axs[1, 1].set_ylabel("acc %")

def experiment_wandb(loader, model):
	pass 


if __name__ == "__main__":
	from dataset_loader import *
	from model_architecture import *
	
	train_loader,test_loader
	model = baseline()
	
	error_func = nn.functional.nll_loss # function is in nn.functional
	
	# optimizer what? & how
	optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.1 )
	
	metrics = Metrics()
	train_model(train_loader, model, error_func, optimizer)
	test_model(test_loader,model,error_func)
	plot_train_test_loss(train_loss_batches, train_acc_batches, test_loss_batches, test_acc_batches)
	
	print("END")