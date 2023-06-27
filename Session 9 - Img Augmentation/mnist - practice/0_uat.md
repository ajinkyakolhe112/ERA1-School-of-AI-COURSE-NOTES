# User Acceptance Testing

### User POV / Functionality
- MNIST, 10 Digits Classification (Problem Statement)
- Web App, where you classify the digit. (Javascript Webpage)
- Types of wrong predictions, Types of high confidence wrong predictions (Understanding Solutions quality)
- Internal Working of Technical Algorithm for added intelligence. (Exceptions made right with human intelligence)

### Technical POV / Implementation
Training Loop

$$
\begin{align*}
& Y_{pred} = model(X_{actual}) \\
error\ =\ loss\ (Y_{pred},Y_{actual}) \\
error.backward() \\
optimizer.step() \\
optimizer.zerograd() \\
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