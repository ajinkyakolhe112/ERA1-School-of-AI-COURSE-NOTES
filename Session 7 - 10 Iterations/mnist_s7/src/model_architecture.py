import torch
import torch.nn as nn
from torch.nn import ReLU as RELU

class S7_Baseline(nn.Module):
	def __init__(self):
		super().__init__()
		self.block1 = nn.ModuleDict({
			"conv1": nn.Conv2d(1,32,8),
			"relu": RELU(),
		})
		self.block2 = nn.ModuleDict({
			"conv1": nn.Conv2d(32,32,8),
			"relu": RELU(),
		})
		self.block3 = nn.ModuleDict({
			"conv1": nn.Conv2d(32,32,8),
			"relu": RELU(),
		})
		self.block4 = nn.ModuleDict({
			"conv1": nn.Conv2d(32,32,7),
			"relu": RELU(),
		})
		self.fc = nn.Linear(32,10)
		
	def forward(self,input_batch):
		b1,b2,b3,b4 = self.block1,self.block2,self.block3,self.block4
		
		b1_output = b1.relu(b1.conv1(input_batch))
		b2_output = b2.relu(b2.conv1(b1_output))
		b3_output = b3.relu(b3.conv1(b2_output))
		b4_output = b4.relu(b4.conv1(b3_output))
		
		b4_output = b4_output.view(-1,32*1*1)
		fc_output = self.fc(b4_output)
		
		output_probs = nn.functional.softmax(fc_output,dim=1)
		output_log = nn.functional.log_softmax(fc_output,dim=1)

		return output_log
	

if __name__ == "__main__":
	test_batch = torch.randn(32,1,28,28)
	model = S7_Baseline()
	test_forward_pass = model(test_batch)
	