inline example: $x$

block example. entire equation needs to be in one line:
$$\ x\ $$

`$$` : Must have empty space before. but first line after can't be empty. 


$$
\begin{align}
Y_{pred} = model(X_{actual})\\
error\ =\ loss\ (Y_{pred},Y_{actual})\\
error.backward()\\
optimizer.step()\\
optimizer.zero\_grad()\\
\end{align}
$$

$$
\begin{align*}
Y_{pred} = model(X_{actual}) \tag{1} \\
error\ =\ loss\ (Y_{pred},Y_{actual}) \tag{2} \\
error.backward() \tag{3} \\
optimizer.step() \tag{4} \\
optimizer.zero\_grad() \tag{5}\\
\end{align*}
$$

$$
\begin{multline*}
p(x) = 3x^6 + 14x^5y + 590x^4y^2 + 19x^3y^3\\ 
- 12x^2y^4 - 12xy^5 + 2y^6 - a^3b^3
\end{multline*}
$$