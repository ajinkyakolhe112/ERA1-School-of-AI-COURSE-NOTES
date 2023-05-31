import torch
import torch.nn as nn
import numpy as np

class Regressor(nn.Module):         # Extending nn.Module, but not overriding forward. Not really using it as a pytorch model.
    def __init__(self, inp_dim=1, out_dim=1, hidden_units=20):
        super(Regressor, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.model = torch.nn.Sequential( # Sequential Container being used in different Class. Sequential & Optmizer are defined inside class. Sequential container is being trained, not Regressor Class
            torch.nn.Linear(inp_dim, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, hidden_units),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_units, out_dim),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def update(self, x, y_target):
        "here dimentions of y_pred and y_target were not matching. Warning was given but didn't read & understand it"
        y_pred = self.model(torch.Tensor(x)).squeeze()
        loss = self.criterion(y_pred, torch.Tensor(y_target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            return self.model(torch.Tensor(x))
        
x = torch.linspace(1, 20, 64)
y = 2 * x + x ** 2 + np.random.rand()
model = Regressor()

epochs = 1000
for epoch in range(epochs):
    model.update(x.unsqueeze(-1), y)