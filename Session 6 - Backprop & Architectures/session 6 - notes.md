One Batch
- Get's sent to single ai accelerator device
- Could get another batch and send to another device. 
- Could get as many devices as one wants

argmax gets index


Architecture: Final Layer
- 3*224*224
- 7*7*512
- 1*1*512*10 or Global Average Pooling

1*1 is superset of FC layer

IMAGE(512,7,7)
GLOBAL AVG POOLING | Just a AVG no discarding |=> (512,1)
FC(512)

OR

IMAGE(512,7,7)
=> 10*512*(1*1) (1*1 Convolution)
GAP

### Code Structure
- `data.py` for "DataSet, DataLoader with Batch"
- `model.py` for " yPred = f(X,W)"
- `utils.py` for "Training / Learning of Model"

Life Cycle of One Batch
- forward pass: y_out = f(X , W)
- errorFun: y_out - y_actual
- finalErrorValueNode: errorValue
- Partial Error wrt W
- TRAIN: Weight Update Acc to Partial Error
    - Batch Details of Error


Good to imagine on Imagenet. Real life objects are easier to visualize.
Imagine a classification problem.
```python
	nn.Sequential( Successive distillation. 
		nn.Conv2d(400, (7,7)), # Localized intelligence. Looks at image, and extracts only information edges & gradients. Or adds edges & gradients info. Extract into channels & combine clustering
		nn.MaxPool2d(2,2)
		nn.Conv2d(400,(7,7)),  # Previous condensed info as input, and further advanced processing
		nn.Conv2d(400,feature maps, (7,7)), # Each Neuron Expected to create, parts of object. How many parts of objects do we have. (Nose of dog). Huge number of sub features
		nn.Conv2d(400 feature maps,(7,7)), 1*1. Each neuron should be, a object relevent to our classes. Dog # imagenet, 10k objects at least. Each with
		nn.Conv2d(10, (1,1)) # Should give us, 10 images we want.
		nn.Linear(10,10),
		nn.Linear(10,10) # 
	)
```
$$
\Large 
\begin{align*}
Conv2d(n\_classes,k = 1*1) \rightarrow Flatten1D &\rightarrow FC(n\_classes)\\
&OR\\
Conv2d(n\_classes, k=1*1) &\rightarrow GAP(full)\\
&OR\\
GAP(full) &\rightarrow FC(n\_classes)\\
\end{align*}
$$
```python
import torch.nn as nn

batch = torch.randn(1,512,7,7)

"GAP -> FC"
nn.AvgPool2d(kernel_size=7) # output: 1*512
nn.Linear(512,n_classes)    # output: 1* n_classes

or
"Conv2d(k=1) -> GAP"
nn.Conv2d(512,n_classes,1) # output: 7*7* n_classes
nn.AvgPool2d(7)     # output: 1* n_classes
```

## Week 6 TODO:
- [ ] Write modular code from scratch for MNIST, CIFAR10, CIFAR100. (3 Modules at least)
- [ ] Explain entire NN using modular code from scratch (2 attempts)
- [ ] Study 5 previous quizzes & their answers. Study 5 lectures
- [ ] Summarize Weekly knowledge and create summary notebook for each week
- [ ] One youtube video in 2 3 weeks