import torch
import torch.nn as nn
from torch.nn import ReLU as RELU

# TODO: Baseline model for Future Customizations.

class S9_Baseline(nn.Module):
	def __init__(m):
		super().__init__()
		m.block1 = nn.ModuleDict({
			"conv1": nn.Conv2d(1,10,(7,7)),
			"relu": RELU()

		})
		m.block2 = nn.ModuleDict({
			"conv1": nn.Conv2d(10,10,(8,8)),
			"relu": RELU()

		})
		m.block3 = nn.ModuleDict({
			"conv1": nn.Conv2d(10,10,(8,8)),
			"relu": RELU()

		})
		m.block4 = nn.ModuleDict({
			"conv1": nn.Conv2d(10,32,(8,8)),
			"relu": RELU()

		})
		m.fc1 = nn.Linear(32,50)
		m.fc2 = nn.Linear(50,10)
		m.log_softmax = nn.functional.log_softmax

	def forward(m,input_batch):
		b1,b2,b3,b4 = m.block1,m.block2,m.block3,m.block4
		b1_output = b1.relu(b1.conv1(input_batch))

		b2_output = b2.relu(b2.conv1(b1_output))

		b3_output = b3.relu(b3.conv1(b2_output))

		b4_output = b4.relu(b4.conv1(b3_output))

		b4_output = b4_output.view(-1,32*1*1) # ! Didn't assign back before

		feature_map = m.fc1(b4_output)
		output_classes_neurons = m.fc2(feature_map)
		output = m.log_softmax(output_classes_neurons, dim=1)

		# output.shape = torch.Size([B, 10])
		return output

if __name__ == "__main__":
	test_data = torch.randn(3,1,28,28) # B C H W
	model = S9_Baseline()
	model(test_data)

	print("END")

