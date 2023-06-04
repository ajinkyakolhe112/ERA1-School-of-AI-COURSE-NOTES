import data
import model
from tqdm import tqdm
import torch.optim as optim
import torch
"Status: DataSet & Model Both Done"
"File Goal: TRAINING & TESTING LOOP"


train_loader, model = data.trainDataLoader, model.model
TO_OPTIMIZE = model.parameters()
lr = 0.01
optimizer = optim.SGD(TO_OPTIMIZE, lr)
training_bar = tqdm(train_loader)
errorFun = torch.nn.functional.nll_loss

def train():
	for batch_no, (x_actual,y_actual) in enumerate(training_bar):
		print("Data is %f",x_actual.shape,"Predicted is ",y_actual.shape)
	
		# $\large Y_{predicted} = model(X_{actual} , W_{layers})$
		y_predicted = model(x_actual)
		errorValue = errorFun(y_predicted,y_actual)
		
		# Recursive calculation of ErrorValue Gradient wrt Weights in model
		errorValue.backward()
		optimizer.step()
		
		training_bar.set_description("Batch %d,\t, Batch Loss %0.3f,\t" % (batch_no, errorValue.item()))
		optimizer.zero_grad()
	
	
	

print("END")