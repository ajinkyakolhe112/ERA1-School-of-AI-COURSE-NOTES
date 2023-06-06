import torch
from tqdm import tqdm
import torch.nn as nn
import data
import model as models
import torch.nn.functional as F

"X , f(x,W), ErrorCalculator, Recursive ErrorGradient, Update, Device"
def trainBatch(X,Y,model,ErrorCalculator,optimizer,device):
    Y_pred = model(X),"builds forward graph of f(X,W)"
    ErrorValue = errorCalculator(Y_pred,Y)
    ErrorValue.backward(),"Calculate Gradient recursively"
    optimizer.step(),"Update Each Parameter W according to Gradient"
    optimizer.zero_grad(),"clear graph for next batch execution"

"X , f(x,W), ErrorCalculator, Recursive ErrorGradient, Update, Device"
def train(train_dataloader,model,errorCalculator,optimizer,device=None):
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
        errorValue.backward()

        "Update W of each neuron recursively"
        optimizer.step()
        for param in model.parameters():
            param.grad # Kernels & Channels

        "batch details"
        trainingProgress.set_description_str("Correctly Predicted %d,\t Error Value %0.4f"%(correct_preds,errorValue.item()))
        print(Y_pred_single,Y_batch)
        optimizer.zero_grad()

"X , f(x,W), ErrorCalculator, Update, Device"
def test_training():

    train_dataloader, test_dataloader = data.train_dataloader,data.test_dataloader
    model = models.model_v1().sequential_model
    ErrorCalculator = torch.nn.functional.nll_loss
    
    lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr)
    train(train_dataloader,model,ErrorCalculator,optimizer)


if __name__=="__main__":
    test_training()

