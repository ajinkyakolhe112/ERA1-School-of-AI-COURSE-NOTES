#%%
import lighning as thor
import torch
import torch.nn as nn

#%%
class MNIST_BASELINE(thor.LighningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=28*28*1,out_features=10)

    def forward(self,image_tensors_batch):
        output = self.fc1(image_tensors_batch)
        output = nn.functional.softmax(output,dim=1)

    def training_step(self, xy_tuple_batch, batch_no):
        image_tensors, image_labels = xy_tuple_batch

        y_predicted = self.forward(image_tensors)
        y_predicted = self(image_tensors)
        error_func = nn.functional.softmax
        error_value = error_func(y_predicted,image_labels)

        return error_value

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(params = self.parameters, lr=0.01)

#%%
model = MNIST_BASELINE()
train_loop = thor.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=3,
)
train_loop.fit(model)
