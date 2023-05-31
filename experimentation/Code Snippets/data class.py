from dataclasses import dataclass
@dataclass
class dataUtils:
    trainLoader: torch.utils.data.DataLoader
    testLoader: torch.utils.data.DataLoader
    trainData: torchvision.datasets.VisionDataset
    testData: torch.utils.data.Dataset
    strDetails: str