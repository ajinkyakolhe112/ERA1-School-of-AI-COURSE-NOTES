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


## Week 6 TODO:
- [ ] Write modular code from scratch for MNIST, CIFAR10, CIFAR100. (3 Modules at least)
- [ ] Explain entire NN using modular code from scratch (2 attempts)
- [ ] Study 5 previous quizzes & their answers. Study 5 lectures
- [ ] Summarize Weekly knowledge and create summary notebook for each week
- [ ] One youtube video in 2 3 weeks