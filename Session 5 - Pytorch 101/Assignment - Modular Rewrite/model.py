import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
"y = f(X,W)"
"PARAMETERS INVESTIGATION, at each layer. Successive transformation of each layer tested if possible or in improving phase"

"Image 3,32,32"
"RF = 32 in last decision layer. 4 Blocks of distillation generally. Which k will have RF 32 =? 33. Which kernel will have RF 32/4 = 8 => 9"


def getModel():
    model = customModel()
    model
    return model

class customModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3,256,(9,9)), nn.ReLU())
        self.block2 = nn.Sequential(nn.Conv2d(256,256,(9,9)), nn.ReLU())
        self.block3 = nn.Sequential(nn.Conv2d(256,256,(9,9)), nn.ReLU())
        self.block4 = nn.Sequential(nn.Conv2d(256,256,(8,8)), nn.ReLU()) # 256*1.0*1.0
        
        self.firstFeaturesReceiver = nn.Sequential(nn.Linear(256*1*1,512), nn.ReLU())
        self.fc2 = nn.Linear(512,10)
        self.finalDecisionMaker = nn.Linear(10,10) # These 10 are Final Neurons for each class for this example
        
    def forward(self, inputData):
        block1_output = self.block1(inputData)
        currentShape = block1_output.shape
        block2_output = self.block2(block1_output)
        currentShape = block2_output.shape
        block3_output = self.block3(block2_output)
        currentShape = block3_output.shape
        block4_output = self.block4(block3_output)
        currentShape = block4_output.shape
        extractedFeatures = block4_output
        currentShape = extractedFeatures.shape
        
        "2D to 1D Conversion. (-1,256) (Batch,TotalOutputPixels) (ndim = 0 & 1). Not (-1,1,256*1*1)"
        extractedFeatures = extractedFeatures.view(-1,256*1*1)
        currentShape = extractedFeatures.shape
        combinedFeatures = self.firstFeaturesReceiver(extractedFeatures)
        currentShape = extractedFeatures.shape
        finalDeactivationActivation = self.fc2(combinedFeatures)
        currentShape = finalDeactivationActivation.shape
        dim = finalDeactivationActivation.ndim
        
        "dim = 1. 0 & 1.. Total 2 dimentions"        
        output = F.softmax(finalDeactivationActivation,dim=1)
        
        logits = torch.randn(2, 6)
        zero = F.softmax(logits,dim=0)
        F.softmax(logits,dim=0).shape
        one = F.softmax(logits,dim=1)
        
        return output
    


    