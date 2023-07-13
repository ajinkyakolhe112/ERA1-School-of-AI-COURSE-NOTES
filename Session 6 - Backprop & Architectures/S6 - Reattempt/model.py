import torch
import torch.nn as nn
import tensorflow as tf
from loguru import logger # !IMP

# input channels can be derived. so just read out channels or neurons first. 
# keras has input channel as must, because it initalizes weights immediately. so it needs to know how many channels are present, those many kernel channels we need.
# !IMP: Architecture Design in Keras is easier
tf.keras.models.Sequential([
    tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 3),
        tf.keras.layers.Conv2D(64, 3), 
        tf.keras.layers.Conv2D(128, 3),
        tf.keras.layers.Conv2D(256, 3),]
    ),
    tf.keras.layers.Conv2D(32, 1),
])

# LAYER PYTHONIC WAY OF WRITING
# layer shuffled for reading of NN. 10 neurons, with (3*3) matrix with 20 channels, operate on 20 channel image data. 
nn.Conv2d(out_channels = 10, kernel_size = 3, in_channels = 20)
# layer for reading according to data. data of 20 channels being transformed into 10 channels with kernel size 3
nn.Conv2d(in_channels = 20, out_channels = 10, kernel_size = 3)

from collections import OrderedDict # !IMP
class BaselineModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kwargs = {  # !IMP
            "padding": "same",
            "stride": 1,
            "bias": False,
        }

        blocks = {
            "b1": nn.ModuleDict({
                "conv1": nn.Conv2d(1,32,3, **kwargs ),
                "relu": nn.ReLU(),
                "conv2": nn.Conv2d(32, 32, 3, **kwargs),
                "relu": nn.ReLU(),
            }),
            "t12": nn.ModuleDict({
                "pool": nn.MaxPool2d(2),
            }),
            "b2": nn.ModuleDict({
                "conv1": nn.Conv2d(32,32,3, **kwargs),
                "relu": nn.ReLU(),
                "conv2": nn.Conv2d(32, 32, 3, **kwargs),
                "relu": nn.ReLU(),
            }),
            "t23": nn.ModuleDict({
                "pool": nn.MaxPool2d(2),
            }),
            "b3": nn.ModuleDict({
                "conv1": nn.Conv2d(32,32,3),
                "relu": nn.ReLU(),
                "conv2": nn.Conv2d(32, 32, 3),
                "relu": nn.ReLU(),
            }),
            "t34": nn.ModuleDict({
                "pool": nn.MaxPool2d(2),
            }),
            "b4": nn.ModuleDict({
                "conv1": nn.Conv2d(32,32,3),
                "relu": nn.ReLU(),
                "conv2": nn.Conv2d(32, 32, 3),
                "relu": nn.ReLU(),
            }),
        }
        # assume same padding. for easier calculations & understanding
        self.features = torch.nn.ModuleDict({
            "b1": nn.Sequential( *blocks["b1"].values() ),
            "t12": nn.Sequential( *blocks["t12"].values() ),
            "b2": nn.Sequential( *blocks["b2"].values() ),
            "t23": nn.Sequential( *blocks["t23"].values() ),
            "b3": nn.Sequential( *blocks["b3"].values() ),
            "t34": nn.Sequential( *blocks["t34"].values() ),
            "b4": nn.Sequential( *blocks["b4"].values() )
        })

    def forward(self, input_batch):
        b1_output = self.features.b1(input_batch)
        logger.debug(f"shape {b1_output.shape}")

        transition_b1_to_b2_output = self.features.t12(b1_output)
        logger.debug(f"shape {transition_b1_to_b2_output.shape}")

        b2_output = self.features.b2(transition_b1_to_b2_output)
        logger.debug(f"shape {b2_output.shape}")

        transition_b2_to_b3_output = self.features.t23(b2_output)
        logger.debug(f"shape {transition_b2_to_b3_output.shape}")

        b3_output = self.features.b3(transition_b2_to_b3_output)
        logger.debug(f"shape {b3_output.shape}")

        transition_b3_to_b4_output = self.features.t34(b3_output)
        logger.debug(f"shape {transition_b3_to_b4_output.shape}")

        b4_output = self.features.b4(transition_b3_to_b4_output)
        logger.debug(f"shape {b4_output.shape}")

        pass

def test_model():
    img = torch.randn(1,28,28)
    model = BaselineModel()
    model(img)

if __name__=="__main__":
    test_model()

