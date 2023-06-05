import torch
import torch.nn as nn

"Sequential or Extending Module"
class base_mark1(nn.Module):
    def __init__(self):
        super().__init__()
        "initialization of layers & parameters here."
        self.first = nn.Conv2d(1,4,kernel_size=(3,3), stride=1, padding=1, bias=True, groups=1) # checking Conv2d in more detail

        self.conv_block1 = self.get_block(4,2) # Delta RF = 8. Effective Kernel Size = 9
        self.relu = nn.ReLU()
        self.converter1 = nn.Conv2d(12,4,3)
        self.conv_block2 = self.get_block(4,2)
        self.relu
        self.converter2 = nn.Conv2d(12,4,3)
        self.conv_block3 = self.get_block(4,2,False) # Output size = 4*4
        self.relu
        self.converter3 = nn.Conv2d(12,4,3)
        self.conv_block4 = self.get_block(4,2,False)
        self.relu

        "depends on output of previous"
        self.combine = nn.Linear(12*6*6,50)
        self.decide = nn.Linear(50,10)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self,Xdata):
        firstOutput = self.first(Xdata)
        secondInput = firstOutput

        "4 Layers, Channels = 4, Padding = 1, Stride = 1, Kernel = 3*3. RF = RF_Delta*Layers + 1"
        block1Features = self.conv_block1(secondInput)
        activatedBlock1 = self.relu(block1Features)
        "outsize = 28,28, channels = 12"
        activatedBlock1 = self.converter1(activatedBlock1)

        furtherFeatures = self.conv_block2(activatedBlock1)
        activatedBlock2 = self.relu(furtherFeatures)
        activatedBlock3 = self.converter2(activatedBlock2)

        furtherFeatures = self.conv_block3(activatedBlock3)
        activatedBlock3 = self.relu(furtherFeatures)
        activatedBlock4 = self.converter2(activatedBlock3)

        furtherFeatures = self.conv_block4(activatedBlock4)
        activatedBlock4 = self.relu(furtherFeatures)

        outputShape = activatedBlock4.shape
        vectorReshaped = activatedBlock4.view(-1,outputShape[1]*outputShape[2]*outputShape[3])

        featureEmbedding = self.combine(vectorReshaped)
        featureEmbedding = self.relu(featureEmbedding)

        decision = self.decide(featureEmbedding)

        finalOutput = self.logSoftmax(decision)
        test = nn.functional.softmax(finalOutput,dim=1)
        return finalOutput
        ""
        
    
    def backward(self,grad_final_leaf):
        ""
    
    "feature extraction block. Single"
    "DIDN'T WORK. KISS"
    def get_block(self,in_channels=32,increament=32,padding=True):
        modules_list = []
        current=in_channels
        for i in range(4):
            layer = nn.Conv2d(current,current+increament,(3,3),padding=int(padding))
            modules_list.append(layer)
            current = current + increament
        block = nn.Sequential(*modules_list)
        # block = nn.Sequential(
        #     nn.Conv2d(32,64,(3,3),padding=1),
        #     nn.Conv2d(64,128,(3,3),padding=1),
        #     nn.Conv2d(128,256,(3,3),padding=1),
        #     nn.Conv2d(256,512,(3,3),padding=1)
        # )
        return block
    

if __name__=="__main__":
    model = base_mark1()
    model(torch.randn(1,1,28,28))