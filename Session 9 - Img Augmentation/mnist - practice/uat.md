# User Acceptance Testing

### User / Functionality
- MNIST, 10 Digits Classification (Problem Statement)
- Web App, where you classify the digit. (Javascript Webpage)
- Types of wrong predictions, Types of high confidence wrong predictions (Understanding Solutions quality)
- Internal Working of Technical Algorithm for added intelligence. (Exceptions made right with human intelligence)

single dollar, inline
$ y = f(X,W) $
double dollar, seperate block but expression in one line while writing
$$  y = f(X,W)$$
math block, allows multi line writing. new line doesn't work. so using a hack
```math\
\begin{aligned}
\Large
Training\ Loop\\
\begin{equation}
Y_{pred} = model(X_{actual})\\
\end{equation}
error\ =\ loss\ (Y_{pred},Y_{actual})\\
error.backward()\\
optimizer.step()\\
optimizer.zero\_grad()\\
\end{aligned}
```

latex block.
```latex
\Large
Training\ Loop\\
Y_{pred} = model(X_{actual})\\
error\ =\ loss\ (Y_{pred},Y_{actual})\\
error.backward()\\
optimizer.step()\\
optimizer.zero\_grad()\\
\\
```
### Technical Implementation
Maths
$\Large Training\ Loop\\ Y_{pred} = model(X_{actual})\\
error\ =\ loss\ (Y_{pred},Y_{actual})\\
error.backward()\\
optimizer.step()\\
optimizer.zero\_grad()\\
\\
$

$$
\Large
Training\ Loop\\
Y_{pred} = model(X_{actual})\\
error\ =\ loss\ (Y_{pred},Y_{actual})\\
error.backward()\\
optimizer.step()\\
optimizer.zero\_grad()\\
\\$$

Architecture / System Design
Fixed Order Base Modules
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