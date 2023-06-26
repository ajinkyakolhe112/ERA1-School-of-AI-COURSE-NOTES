RF Formula in Excel & Markdown Latex Equation

## With Padding & Stride
$$
\Large
n_{output} = \frac {(n_{input} + 2 \cdot p - k)}{s} + 1\\

rf_{output} = rf_{input} + (k - 1) \cdot s \cdot j_{input}\\

j_{output} = j_{input} \cdot s
$$

## Without Padding
$$

\Large
n_{output} = \frac{n_{input} - k}{s} + 1\\

rf_{output} = rf_{input} + (k - 1) \cdot s \cdot s_{accum}\\

depth = \frac {rf_{goal}} {(k -1) \cdot s \cdot s_{accum}}\\

\Delta rf = (k-1) \cdot s \cdot s_{accum}\\

$$
