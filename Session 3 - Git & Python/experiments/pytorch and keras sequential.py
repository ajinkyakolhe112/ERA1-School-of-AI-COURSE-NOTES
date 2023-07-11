import torch
import torch.nn as nn

from torchinfo import summary
from torchvision import datasets,transforms
from sklearn import datasets as toysets

testScalar = torch.tensor(10) # a cell
testVector = torch.randn(10) # (10,1)
testImage = torch.randn(10,10)
testImageTensor = torch.randn(10,28,28) # 2 dimentional h*w image in c channels. Channel is 3rd dimention

class customNN(nn.Module):
	def __init__():
		super().__init__()
		self.conv1 = nn.Conv2d(input_channels,output_channels,(3,3))
	
	def forward(self,input_data_batch):
		first_layer_output = self.conv1(input_data_batch) # CHW
		output = F.relu(first_layer_output)
		pass


model = keras.Sequential(
	layers.Dense(units=1,activation="relu"),
)
y_pred = model(x_train_data)
error_func = keras.losses.MeanSquaredError()
optimizer = keras.optimizer.Adam(learning_rate = 0.001)
model.compile(loss=error_func, optimizer = optimizer)
model.fit(x_train_data,y_train_data)

print("End")