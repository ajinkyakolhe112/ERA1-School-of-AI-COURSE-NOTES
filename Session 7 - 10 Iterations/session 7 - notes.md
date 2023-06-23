
## MNIST Dataset
```python
dataset = torchvision.datasets.MNIST()

print(dataset)
"""
Dataset MNIST
    Number of datapoints: 60000
    Root location: ./data
    Split: Train
    StandardTransform
Transform: Compose(
               ToTensor()
               Normalize(mean=(0.1307,), std=(0.3081,))
           )
"""

len(dataset)	# dataset is list of 60k elements of data points
len(dataset[0])	# dataset[0] is a tuple of length 2. each datapoint is data & its value

# Dataset is a list of elements & value of the element.
for data,target in datasets:
	print(data.shape,target)

data, targets = dataset.data, dataset.targets

```

## Imp Functions

- **`data`**: list of data (not their values)
- **`targets`**: list of target values
- `classes`: list prediction classes in dataset
- `transform`: transform on data
- `target_transforms`: transforms on target
- ~~`transforms`~~: combination of `transform` & `target_transform`

Deprecated
- ~~`train_labels`~~
- ~~`test_labels`~~
- ~~`train_data`~~
- ~~`test_data`~~
- `training_file` & `test_file`: Not much used

## Source Code for MNIST & Vision Datasets
- https://github.com/pytorch/vision/blob/main/torchvision/datasets/vision.py
- https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py
- Imp Methods. `__init__()`, `__getitem__()` & `__len__()`
	- Both have `raise NotImplementedError`
	- https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py#L138-L148
	- https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py#L152-L153

## Notes on Training Graph
- Loss Value is **Batch Loss Value**. If we calculate Error value of same batch, error would reduce. But we move to next batch directly
- Accuracy is **Total Accuracy not Batch Accuracy**.
- Batch Size, Image Size, DIKW Feature Density in Image
- Total Object & generalization, Parts of that Object & generalizations from huge DATA
- Number of Parameters in Memory, Multiplications in Convs, 
- Depth, Gradient at depth
- Learning Rate is important


# Data & Model
Pattern Complexity: Low, Medium, High
Network Complexity: Low, Medium, High

Overfitting  ??
Underfitting ??

----
---
Principles of Pytorch
1. Low Level Library & Pythonic. (Not comprehensive functions because low level. Just essential. Pythonic: Easy to Read Source Code)
- Need to be familiar with its source code, because core essential functionality is given. Understand what is given & what is not given. Use external libraries or write own library
- torchsummary or torchinfo
- train & test methods in pytorch
- Because Pythonic, its easier to read
- Why Low Level??
- Field is Nascent. New Knowledge needs to be implemented so library should be flexible
- Hardware for running library: Nvidia, Cuda. New: Apple, AMD
- Librares Optimized for Hardware. 
- Pytorch
- Julia or Jax for accelerated performance. Need to write things from scratch where library isn't built
2. Not just using Pytorch API, but reading source code & understanding how to do software engineering of Python + Forum + Podcast, together should give an deeper understanding of Pytorch
3. Writing Good README & Essential Functions & Fun Name Design. (Verbose - Succinct)
4. Don't overfit on Pytorch. Generalize on Library not Overfit on Pytorch.
1. Because Pytorch is so popular, overtime knowledge of NN becomes same as Knowledge of Pytorch.
2. So think in terms of general NN always, and then convert that to Pytorch. That way, knowledge is generalizable across libraries. 