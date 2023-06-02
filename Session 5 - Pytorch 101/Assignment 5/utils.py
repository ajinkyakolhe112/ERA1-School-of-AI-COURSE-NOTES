from tqdm import tqdm

# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def GetCorrectPredCount(YPredicted, YActual):
	yFinalClass = YPredicted.argmax(dim=1)
	comparison = yFinalClass.eq(YActual)
	count = comparison.sum().item()
	return count

def train(train_loader,model,errorFun, optimizer, device):
	model.train()
	
	train_loss = 0
	totalCorrect = 0
	processed = 0
	trainProgressBar = tqdm(train_loader)
	
	# XDATA: (B,C,H,W)
	for batch_id, (XData, YActual) in enumerate(trainProgressBar):
		XData.to(device),YActual.to(device)
		
		YPredicted = model(XData)
		errorValue = errorFun(YPredicted,YActual)
		errorValue.backward() # loss backprop in this batch
		"Gradients of W, according to Contribution in Error"
		
		optimizer.step() # weight update in this batch
		optimizer.zero_grad()
		
		train_loss += errorValue.item() # train loss in this batch
		totalCorrect += GetCorrectPredCount(YPredicted,YActual) # correct values in this batch
		processed_images += XData.shape[0]
		trainProgressBar.set_description("Batch No %f, Processed Images %f, Error Value %f, Accuracy Value %f" % (batch_id,processed_images, train_loss,totalCorrect) )

	train_acc.append(100*totalCorrect/processed)
	avg_training_loss = train_loss/len(train_loader)
	train_losses.append(avg_training_loss)

def test(test_loader, model, errorFun, device):
	model.eval()
	
	test_loss = 0
	correct = 0
	
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(test_loader):
				data, target = data.to(device), target.to(device)
				output = model(data)
				test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
				correct += GetCorrectPredCount(output, target)
	
	test_loss /= len(test_loader.dataset)
	test_acc.append(100. * correct / len(test_loader.dataset))
	test_losses.append(test_loss)
	
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
			test_loss, correct, len(test_loader.dataset),
			100. * correct / len(test_loader.dataset)))