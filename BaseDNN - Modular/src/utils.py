from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as torch_optimizer

def train(train_loader, model, errorFun, optimizer, epoch_no, device=None):
	model.train()
	model.to(device)
	pbar = tqdm(train_loader)
	for batch_idx, (data, target) in enumerate(pbar):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = errorFun(output, target)
		loss.backward()
		optimizer.step()
		pbar.set_description("loss= %f,     batch_id= %d" % (loss.item(),batch_idx))
#		pbar.set_description("loss= {0},    batch_id= {1}   ".format(loss.item(),batch_idx))
#		pbar.set_description("loss= {0.2f}, batch_id= {0.2f}".format(loss.item(),batch_idx))

def test(testDataLoader, model, errorFun, device=None):
	model.eval()
	test_loss_total = 0
	correct_preds_total = 0
	processed_total = 0
	pbar = tqdm(test_loader)
	with torch.no_grad():
		test_loss_batch = 0
		correct_preds_batch = 0
		for data, target in pbar:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss_batch = errorFun(output, target, reduction='sum').item()   # sum up batch loss
			preds_batch = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct_preds_batch += pred.eq(target.view_as(preds_batch)).sum().item()
			
			test_loss_total += loss_batch
			correct_preds_total += correct_preds_batch

			count += 1
			pbar.set_description("batch loss = %f\t,batch correct = %d\t,batch accuracy %f",(test_loss,correct,target))
	
	test_loss /= count
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
		format(test_loss, correct, count,100. * correct / count))

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
	
	

