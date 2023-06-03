import torch
import torch.nn as nn

# Linear means ndim = 1. of single example

# Fake Dummy Data of Single Example
examples = 10
featureColumns = 6 # 6 columns in a row
inputData = torch.rand(examples, featureColumns)
yObserved = torch.rand(examples,1)

inputData.ndim

class customLinear(nn.Module):
  def __init__():
    super().__init__(weightDimentions=0,transformOperation=0)
    self.customWeights = weightDimentions # This doesn't initialize weight values actually. of weight dimentions, there needs to be an array. 
    self.customWeights = customFunction.getDataRandom(shape) # This would also not work because, pytorch will say, no parameters found
    self.customWeights = nn.Parameters(shape,dtype=float32,requires_grad=True)
    self.whateverDefinedParameter = nn.Parameters() # THIS PARAMETER can be set to learning. Will make learning slower but can get a better value.
    self.transformOperation = transformOperation
    # Here we could have a powerValueLayer
    # at its core, what is the operation. How many income connections.. That is number of feature columns. 
    # what is the operation defined as, it could require same numner of weights or it could require some different. 
    # transformedX = X operation W
  
  def forward(self,inputData):
    print(self.inputData, self.transformOperation, self.customWeights)
    transformedInput = inputData @ self.customWeights
    return transformedInput

customLinear = customLinear(featuresIncoming,"because element wise dot operation, same no of weights on neuron", numberOfNeurons)

nn.Sequential(
    nn.Linear(featureColumnsIncoming,4), # LinearDimData_DotProductOperation Layer
)