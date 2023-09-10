
Deep Learning & NN Field is about just 4 things.
1. Data & its representation. **any kind of Data -> Tensor**. (Tensor -> Pytorch, Numpy, Tensorflow, or any other programming language)  
   1. $\large data = (X_{collected}, Y_{collected})$
   2. $\large Data \rightarrow Information \rightarrow Knowledge about Patterns \rightarrow Understanding of Patterns \rightarrow Wisdom$
   3. **Huge Data**, Compress, Condense, Fundamental Patterns Extraction, Generalization, **Smallest essential logical building blocks**
2. Neural Network & it's parameters $\Large W$. as Deep Function Approximator & Mathematical Learning
   1. Neuron, Group of Neurons = Layers, Group of Layers = Block of Layers
   2. `Training Loop`
      1. $\large model = f(X,W_{[layer\_no][neuron\_no]})$
      2. $\large Y_{predicted} = f(X_{collected},W_{[layer\_no][neuron\_no]})$
      3. $\large error\_value = error\_func(Y_{predicted}, Y_{collected})$
      4. $\huge minimize(error\_value)$
3. Hardware running the NN
4. Learning Algorithm running on Hardware

Receptive Field of Image & Attention

Experiment
1. Image Data, Matrix & Tensor
2. Channels of Data
3. Neuron or Layer or Block or Network = Transformations (PYTORCH)
   1. Input
   2. Output
   3. Transform: (Geometry + Matrix Multiplication + Latex Tikz & Latex + Javascript webpage)
   3. Parameters in Layer
      1. Parameter count 
      2. Parameter memory
   4. Flops Operations
4. DIKW Successive Transformations & Pytorch transformations
   1. Pytorch model as sequential
5.  Types of Layers. Conv, FC (PYTORCH)
   1. [Channel Notes & Experiments colab](https://colab.research.google.com/drive/1cKqF4fO5eWTOITeeFfe5Oy0fERGmiuML)
6. Python Memory Profile & Python Execution Profile



Image Data = (Channels, Height Pixels, Width Pixels)
**Color Depth = 8bits color depth**
- Image Dataset format is: (Number of Examples, Number of Channels, Height Pixels, Width Pixels)
- 1 Pixel = Total number of shades of color. Color Depth = bits per color. 
- Human Vision Color Depth = 8bits per color. We see 3 such colors, Red, Green & Blue

https://en.wikipedia.org/wiki/Color_depth
- Human Eye   = 10 million colors. 
- TV          = 24 bits for rgb = 2^24 = 16.7 million colors
- TV          = 30 bits for rgb = 2^30 = 1.07 billion colors = 64 times more colors than 24 bit

vs "Total Pixels with that Color Depth"
- Human Eye       = Total Simulated Megapixels 576. Actual megpixels = 7 mega pixels at center
- TV Resolution   = 1920 * 1080 pixels (1080p)    = 2    mega pixels 
- TV Resolution   = 3840 * 2160 pixels (4K)       = 8.3  mega pixels
- TV Resolution   = 7680 * 4320 pixels (8K)       = 16.3 mega pixels

Colored Image = Color Channel * Height Pixels * Width Pixels
Each Pixel    = 8bit Color Depth Color