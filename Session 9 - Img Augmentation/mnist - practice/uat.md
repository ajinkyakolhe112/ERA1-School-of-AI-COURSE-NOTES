# User Acceptance Testing

### User / Functionality
- MNIST, 10 Digits Classification (Problem Statement)
- Web App, where you classify the digit. (Javascript Webpage)
- Types of wrong predictions, Types of high confidence wrong predictions (Understanding Solutions quality)
- Internal Working of Technical Algorithm for added intelligence. (Exceptions made right with human intelligence)


### Technical Implementation
Maths
$$
\Large
Training\ Loop\\
Y_{pred} = model(X_{actual})\\
error\ =\ loss\ (Y_{pred},Y_{actual})\\
error.backward()\\
optimizer.step()\\
optimizer.zero\_grad()\\
\\
$$

Architecture / System Design
- `dataset_load.py` - Doesn't change much
- `train_test.py` - Doesn't change much
- `model_dev.py` - Dev Copy for model improvement. Future experimentation here
- `models.py` - Saving All Models here. Increamental 
- `notebook.ipynb` - experiment executed here
- `notebook.py` - experiment in script format



