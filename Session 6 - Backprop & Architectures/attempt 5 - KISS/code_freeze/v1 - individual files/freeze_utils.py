def train(train_loader, model, error_func, optimizer, epoch_no, device=None):
	
	model.to(device)
	pbar = tqdm(train_loader)
	
	train_loss_total = 0
	correct_preds_total = 0
	for b_id, (images, labels) in enumerate(pbar):
		images,labels = images.to(device), labels.to(device)
		
		#Forward pass
		outputs = model(images)
		error_batch = error_func(outputs, labels,reduction='mean')
		
		# Backward and optimize
		optimizer.zero_grad()
		error_batch.backward()
		optimizer.step()
		
		pred_labels = outputs.argmax(dim=1, keepdim=True).squeeze()
		correct_preds_batch = torch.eq(pred_labels, labels).sum().item()
		accuracy_batch = 100.0 * correct_preds_batch / BATCH_SIZE
		pbar.set_description("Epoch %d, Batch= %d, Batch Error Value = %f, Batch Pred Acc = %f" % (epoch+1,b_id+1, error_batch.item(), accuracy_batch))
		correct_preds_total = correct_preds_total + correct_preds_batch
		train_loss_total = error_batch.item()
		if (b_id + 1) % 100 == 0:
			print("Epoch %d, Batch %d, Error Value = %f, Pred_Acc = %f" % (epoch+1,b_id+1, error_batch.item(), accuracy_batch))
	train_losses[epoch_no] = train_loss_total / len(train_loader)
	train_accuracy[epoch_no] = 100.0 * correct_preds_total / len(train_loader)
	print("Epoch %d, Final Accuracy of the network %f" % (epoch +1 , train_accuracy[epoch_no]))

def test(test_loader, model, error_func,epoch, device=None):
	torch.set_grad_enabled(False)
	
	test_loss_total = 0
	correct_preds_total = 0
	pbar = tqdm(test_loader)
	for b_id,(images, labels) in enumerate(pbar):
		images,labels = images.to(device), labels.to(device)
		
		outputs = model(images)
		error_batch = error_func(outputs, labels, reduction='mean')
		
		pred_labels = outputs.argmax(dim=1, keepdim=True).squeeze()
		correct_preds = torch.eq(pred_labels, labels).sum().item()
		accuracy_batch = correct_preds / BATCH_SIZE * 100
		
		test_loss_total = test_loss_total + error_batch.item()
		correct_preds_total += correct_preds
		
		pbar.set_description("Epoch %d, Batch= %d, Error Value = %f, Pred Acc = %f" % (epoch+1, b_id+1, error_batch.item(), accuracy_batch))
		if (b_id + 1) % 100 == 0:
			print("Epoch %d, Batch %d, Error Value = %f, Pred_Acc = %f" % (epoch+1,b_id+1, error_batch.item(), accuracy_batch))
	test_losses[epoch] = test_loss_total / len(train_loader)
	test_accuracy[epoch] = 100.0 * correct_preds_total / len(test_loader)
	print("Epoch %d, Final Accuracy of the network %f" % (epoch +1 , 100.0 * correct_preds_total / len(test_loader)))
	torch.set_grad_enabled(True)