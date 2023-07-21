import torch
import torch.nn
import pytorch_lightning as thunder

# Model Architecture & Forward Method
class Baseline(nn.Module):
    def __init__(self):
        pass

    def forward(self,x_batch):
        pass

# Extended: Training Loop, Optimization on top of existing
class BaselineExtended(thunder.LightningModule):
    def __init__(self, model, loss):
        super().__init__()
        self.automatic_optimization = False
    
    def configure_optimizers():
        optimizer = torch.optim.Adam(self.parameters(), lr= 0.001)

    def forward(self, batch):
        pass

    def training_step(self, batch, batch_no):
        x_train, y_train = batch
        optimizer = self.optimizers()
        
        optimizer.zero_grad()
        y_predicted = self(x_train)
        loss = self.error_func(y_predicted, y_train)
        self.manual_backward(loss)
        optimizer.step()

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
        pass

    def optimizer_step(self, optimizer, batch_idx, epoch_idx, optimizer_idx):
        optimizer.step()
        pass    
    

train_loader = 0

model_extended = BaselineExtended()
model_trainer = thunder.Trainer()
model_trainer.fit(super_model, train_loader)