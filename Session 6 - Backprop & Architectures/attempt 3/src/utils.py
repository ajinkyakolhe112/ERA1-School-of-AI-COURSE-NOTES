import torch
from tqdm import tqdm
import torch.nn as nn
import data
import model as models
import torch.nn.functional as F

class reshape(nn.Module):
    def __init__(self,pixels):
        super().__init__()
        self.pixels = pixels
    def forward(self,x):
        x = x.view(-1,512*9)
        return x
reshaper = reshape(512*9)
test_model_outside = nn.Sequential(*{                          # RF Start = 1
            # img = [ in_channels, out_channels ] , kernel = [ kernel = (3,3), stride , padding ]
            "conv1": nn.Conv2d(1,32,(3,3),1,1),     # Delta RF = +2
            "relu": nn.ReLU(), 
            "conv2": nn.Conv2d(32,64,(3,3),1,1),    # Delta RF = +2
            "relu": nn.ReLU(),
                                                    # kernel = [ kernel = (3,3), stride , padding ]
            "pool1": nn.MaxPool2d((2,2),2,0),       # Delta RF = *2

            "conv3": nn.Conv2d(64,128,(3,3),1,1),   # Delta RF = +2
            "relu": nn.ReLU(),
            "conv4": nn.Conv2d(128,256,(3,3),1,1),  # Delta RF = +2
            "relu": nn.ReLU(),
                                                    # kernel = [ kernel = (3,3), stride , padding ]
            "pool2": nn.MaxPool2d((2,2),2,0),       # Delta RF = *2. RF Total = 28
            
            "conv5": nn.Conv2d(256,512,(3,3),1,0),  # Delta RF = +2
            "relu": nn.ReLU(),
            "conv6": nn.Conv2d(512,512,(3,3),1,0),  # Delta RF = +2, Output = (K:512,H:3,W:3)
            "relu": nn.ReLU(),
            "reshape": reshaper,

            # "merge": nn.Conv2d(512,1,(1,1),1,0),    # Reducing Channels to 1. So that we get [1,3*3]. Shape for Linear
            # "flatten": nn.Flatten(),                # RF Total = 32, Output = (512,9)
            
            "fc1": nn.Linear(512*9,50),               # Output = 512,10
            "relu": nn.ReLU(),
            "fc2": nn.Linear(50,10),                # 10 Neurons for 10 Classes. Output = [1*10]
            "softmax": nn.LogSoftmax(dim=1),           # dim0 = 1, dim1= 10 in Output = (1,10)
            # Using softmax to check if probability sum == 1

        }.values())
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
        errorValue.retain_grad()
        errorValue.backward()


        "Update W of each neuron recursively"
        # optimizer.step()
        for param in model.parameters():
            delta = param.grad * 0.01
            param = param - param.grad * 0.01
            # delta = param.grad * 0.01 . ERROR. Done AFTER param is updated. Hence grad depopulated

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

    "custom test model injected here"
    train(train_dataloader,test_model_outside,ErrorCalculator,optimizer)


if __name__=="__main__":
    test_training()

