import torch
import pytorch_lightning as thor

import loguru.logger
import wandb

x_train, y_train
x_val, y_val

class AbstractNN(nn.Module):
    def __init__(self):
        self.model = nn.ModuleDict({
            "l1", nn.Linear(784,50),
            "l2", nn.ReLU(),
            "l3", nn.Linear(50,10),
            "l4", nn.Softmax(10)
        })

    def forward(self, x_batch):
        f = self.model
        l1,l2,l3,l4 = f.l1,f.l2,f.l3,f.l4

        l1_output = l1(x_batch)
        l2_output = l2(l1_output)
        l3_output = l3(l2_output)
        l4_output = l4(l3_output)

        x = l4(
                l3(
                    l2(
                        l1(
                            x
                        )
                    )
                )
            )
        
        return l4_output

class ThorNNImproved(thor.LightningModule):
    def __init__(self, model: AbstractNN):
        self.nn = model

        pass
    
    def train_step(self, x_batch):
        pass
    
