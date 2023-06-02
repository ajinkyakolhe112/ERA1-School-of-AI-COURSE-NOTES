# Assignment 5

## Usage
- Run the notebook `S5.ipynb` to train & test the model
- Model Architecture in `model.py`

---

## Solution Details
### Modules
- `model.py`: classes representing the convolutional neural network
- `utils.py`: helper functions for the training engine
  - `train`: train for one epoch on the training set
  - `test`: test using the testing set

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
            Conv2d-2           [-1, 64, 24, 24]          18,432
         MaxPool2d-3           [-1, 64, 12, 12]               0
            Conv2d-4          [-1, 128, 10, 10]          73,728
            Conv2d-5            [-1, 256, 8, 8]         294,912
         MaxPool2d-6            [-1, 256, 4, 4]               0
            Linear-7                   [-1, 50]         204,800
            Linear-8                   [-1, 10]             500
================================================================
Total params: 592,660
Trainable params: 592,660
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.77
Params size (MB): 2.26
Estimated Total Size (MB): 3.03
----------------------------------------------------------------  
```