import torch
from torch import nn

# for image channels are last. because command sense image dimentions are 
class PyTorchModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            # input:       256*256* 3 , 1 conv output: 64*64* 65
            # kernel:      1 kernel = 11*11 *3. "65" kernels: 11*11*3 * 65
            nn.Conv2d(3, 65, kernel_size=11, stride=4, padding=2),
            # features:     Feature Map Size / Feature Map / Channel Size = 64*64* 65. Channels = 65
            # GLOBAL RF: Delta RF = (kernel - 1) * stride * j_accum. Total RF = 1 + 40. 
            
            nn.ReLU(inplace=True),
            # discarded pixels = 50%. Decision made, threshold defined for wrong
            
            # maxpool: Delta RF: (kernel - 1) * stride * j_accum = 16. Total RF = 41 + 16 = 57
            # discarded pixels. 1 out of 9. Means its okay to discard. Must mean 11*11 is imp kernel size
            # Stride = 2 < kernel. Some overlap, but some images are covered only once
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 1. padding for same channel size
            # 2. max depth with simple kernel size
            # 3. channel numbers
            # 3. channel size
            # 4. global RF calculation
            # 5. effective kernel size until that level

            nn.Conv2d(65, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits

model = PyTorchModel(10)
test_img = torch.randn(3,28,28)

model(test_img)
