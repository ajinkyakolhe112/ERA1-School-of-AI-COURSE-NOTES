# Problem
- `MNIST` Data
- `less than 20k Parameters` Model Constraint
- `less than 20 epochs` Training Constraint

1. `data.py`
    - $\Large X_{batch_{i}} \ , Y_{batch_{i}}$
    - $i = 0, 1, 2 \ldots, batches_{no}$
2. `model.py`
    - $\Large X = \vec{[ x_{0},x_{1},\ldots,x_{pixels}]}$
    - $\Large model = f(X_{batch_{i}} , W ) $
    - $\Large model = f(X_{batch_{i}} , W_{[layers]}\ _{[neuron\_id]} ) $

### Building Solution Plan
Data X, Model f(x,W), Forward Pass f(x,W)(X), Error & Reducing Error & Testing at End

$$
\displaystyle
\large X_{i} = data \hspace{100cm}\\
\large f([x_{i}],W) = model \hspace{100cm}\\

\large Y_{out} = f(X_{batch\_no},W_{}{}) \hspace{100cm}\\
\large E = Y_{out} - Y \hspace{100cm}\\
\large W = W - \frac{\partial{E}}{\partial W_{i}} * lr


$$
