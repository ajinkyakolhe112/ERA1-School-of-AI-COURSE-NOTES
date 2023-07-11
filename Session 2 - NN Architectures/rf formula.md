RF Formula in Excel & Markdown Latex Equation
```python
height,width,channels = (28,28,1)
pytorch_img = torch.randn(channels,height,width)

conv_operation = nn.Conv2d(in_channels, out_channels, (k,k), stride= 1, padding= 0)
# arguments are in descending order of their importance. (in_channels, out_channels) & (kernel,stride,padding)
# main is kernel size. nested across depth times in block of layers

# padding is choosen for keeping channel size same. Its difficult to manually remember what's the channel size which changes each layer. so better choose a padding value, so that, channel size remains the same. 
# stride for increasing jump size. reduces channel size drastically. its introduced occasionally

# !IMP: GOOGLE SHEETS, with analysis of kernel, stride, padding. and important extra properties we should know.
```
$$
\begin{align*}
img &= (H,W,C)\\
kernel &= (k,k,C)\\
output &= (reduced, reduced, 1)\\
\\
height_{output} &= \frac {(height_{input} - kernel + 2 \cdot padding )}{stride} + 1 \tag{1}\\
\end{align*}
$$


$$
\begin{align*}
rf^{global} &= rf_{input} + (kernel - 1) \cdot stride \cdot s_{accum}\\
\\
s_{accum} &= s_{accum} \cdot stride\\
\\
depth &= \frac {width} {(kernel -1) \cdot s \cdot s_{accum}}
\end{align*}
$$

Stride = 1, padding for same height
$$
\begin{align*}
s &= 1 \\
padding &= \frac{( kernel - 1 )}{2}\\
\Delta rf &= kernel - 1\\
depth &= \frac{width}{kernel - 1}\\
\end{align*}
$$
