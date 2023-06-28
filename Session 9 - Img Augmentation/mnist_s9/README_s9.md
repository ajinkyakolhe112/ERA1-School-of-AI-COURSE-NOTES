# User Acceptance Testing
**Agenda**
1. User POV / Functionality
2. Technical POV / Implementation
3. Scientist POV / Experiment Log

### 1. User POV / Functionality
- MNIST, 10 Digits Classification (Problem Statement)
- Web App, where you classify the digit. (Javascript Webpage)
- Types of wrong predictions, Types of high confidence wrong predictions (Understanding Solutions quality)
- Internal Working of Technical Algorithm for added intelligence. (Exceptions made right with human intelligence)

### 2. Technical POV / Implementation
Simple Training Loop

$$
\begin{align*}
Y_{pred} = model(X_{actual}) \tag{1} \\
error\ =\ loss\ (Y_{pred},Y_{actual}) \tag{2} \\
error.backward() \tag{3}\\
optimizer.step() \tag{4}\\
optimizer.zerograd() \tag{5}\\
\end{align*}
$$

Architecture / System Design: Base Modules (Fixed Order)
- `dataset_load.py` - Doesn't change much
- `model_dev.py` - Dev Copy for model improvement. Future experimentation here
- Blocks with nn.ModuleDict for easier calling layers for forward pass
- If done with nn.Sequential then easier to get layer output by calling layer index
- `train_test.py` - Doesn't change much. 
- Function definitions for consistency
	- `def train_model(train_loader, model, error_func, optimizer, device=None, epoch_no=1):`
	- `def test_model(test_loader, model, error_func, device=None, epoch_no=1):`
	- **Metrics** to be monitor equally important as train & test methods
- `utils.py` - General Utils. Experiment Tracking, Logging, Visualization
- `notebook.ipynb` - experiment executed here
Later
- `models.py` - Saving All Models here. Increamental 
- `notebook.py` - experiment in script format


### 3. Scientist POV / Experiment Log
1. ~~`Session 9 Model`~~ doesn't learn. But `Session 7 Model` does. Some bug in code
   - [x] tested S7 model, doesn't learn here. But same model learns in S7. Bug in dataset or train
   - [x] test S9 model, with S7 training. It works. Bug in likely in `s9_train_test.py`
   - [x] exact same architecture in both, and still not working