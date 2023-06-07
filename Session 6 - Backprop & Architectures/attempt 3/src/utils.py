import torch
from tqdm import tqdm
import torch.nn as nn
from data import *
from model import *
import torch.nn.functional as F

"X , f(x,W), ErrorCalculator, Recursive ErrorGradient, Update, Device"
def trainBatch(X,Y,model,ErrorCalculator,optimizer,device=None):
    Y_pred_probs = model(X)                       # builds forward graph of f(X,W)
    ErrorValue = ErrorCalculator(Y_pred_probs,Y)  # Calculate Error Value
    ErrorValue.backward()                   # Calculate Gradient recursively
    # optimizer.step()                      # Update Each Parameter W according to Gradient
    for name,param in model.named_parameters():
            delta = param.grad * 0.01
            param = param - param.grad * 0.01
            # delta = param.grad * 0.01 . ERROR. Done AFTER param is updated. Hence grad depopulated
    optimizer.zero_grad()                   # clear graph for next batch execution"

    print("Predicted Value",Y_pred_probs.argmax(dim=1),"Actual Value",Y,"Single Error Value",ErrorValue.item())
    # print("Predicted Value",Y_pred_probs.argmax(dim=1),"Predicted Probs",Y_pred_probs)

"X , f(x,W), ErrorCalculator, Recursive ErrorGradient, Update, Device"
def trainEpoch(train_dataloader,model,errorCalculator,optimizer,device=None):
    model.train(mode=True)
    trainingProgress = tqdm(train_dataloader)
    
    for batch_no, (X_batch,Y_batch) in enumerate(trainingProgress):
        X,Y_actual = X_batch,Y_batch
        
        # Batch dim0,n_classes dim1
        Y_pred_probs = model(X)
        # Batch
        Y_pred_single = Y_pred_probs.argmax(dim=1)
        "Error Value for model"
        errorValue = errorCalculator(Y_pred_probs,Y_actual)
        "Acc Value for Humans"
        correct_preds = torch.eq(Y_pred_single, Y_actual).sum().item()


        "Calculate dE/dW for each neuron recursively"
        errorValue.retain_grad()
        errorValue.backward()


        "Update W of each neuron recursively"
        # optimizer.step()
        for param in model.parameters():
            non_zero_grads = param.grad.count_nonzero()
            param = param - param.grad * 0.01
            # delta = param.grad * 0.01 . ERROR. Done AFTER param is updated. Hence grad depopulated

        "batch details"
        trainingProgress.set_description_str("Correctly Predicted %d,\t Error Value %0.4f"%(correct_preds,errorValue.item()))
        # print(Y_pred_single,Y_batch)
        optimizer.zero_grad()

def trainEpochs(train_dataloader,model,errorCalculator,optimizer,epochs=10,device=None):
    for epoch_no in range(epochs):
        trainEpoch(train_dataloader,model,errorCalculator,optimizer)

"X , f(x,W), ErrorCalculator, Update, Device"
def test_training():

    train_dataloader, test_dataloader
    model = model_v1().sequential_model
    ErrorCalculator = torch.nn.functional.nll_loss
    
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr)

    batch_no = 0
    epochs = 1
    while epochs < 20:
        iterator = iter(train_dataloader)
        while batch_no < 50:
            X,Y = next(iterator)
            trainBatch(X,Y,model,ErrorCalculator,optimizer)
            batch_no += 1
        epochs += 1


if __name__=="__main__":
    test_training()

