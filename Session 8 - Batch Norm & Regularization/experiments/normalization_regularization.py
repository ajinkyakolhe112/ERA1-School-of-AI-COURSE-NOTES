import torch
import torch.nn as nn # works if nn is module. doesn't if nn is a function
import pytorch_lightning as thor


layer_filters = (1,32,64,128,256)
kernel_size = 7,7
compress = 1,1
# sequential is a great building block, because you don't need to write its forward
feature_extractor = nn.ModuleDict({
    "conv1": nn.Conv2d(layer_filters[0], layer_filters[1], k = kernel_size),
    "conv2": nn.Conv2d(layer_filters[1], layer_filters[2], k = kernel_size),
    "conv3": nn.Conv2d(layer_filters[2], layer_filters[3], k = kernel_size),
    "conv4": nn.Conv2d(layer_filters[3], layer_filters[4], k = kernel_size),

    "conv5": nn.Conv2d(layer_filters[4], 10*15, compress),
    "conv6": nn.Conv2d(10*15, 10*15, (3,3)),
    "conv7": nn.Conv2d(10*15, 10, compress),
    "gap":   nn.GlobalAveragePooling2d(),
    "fc1":   nn.Linear(10,10),
})

layers_experiment = nn.ModuleDict({
"batch_norm": nn.BatchNormalization(neurons[1]), 
"layer_norm": nn.LayerNormalization(),
"group_norm": nn.GroupNormalization(),
})

