
Deep Learning & NN Field is about just 4 things.
1. Data & its representation. **any kind of Data -> Tensor**. (Tensor -> Pytorch, Numpy, Tensorflow, or any other programming language)  
   1. $\large data = (X_{collected}, Y_{collected})$
   2. $ Data \rightarrow Information \rightarrow Knowledge\ about\ Patterns \rightarrow Understanding\ of\ Patterns \rightarrow Wisdom$
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

## Color Depth
https://en.wikipedia.org/wiki/Color_depth
- Image Dataset format is: (Number of Examples, Number of Channels, Height Pixels, Width Pixels)
- 1 Pixel = Total number of shades of color. Color Depth = bits per color. **Color Depth** = Color Resolution
- Each Pixel    = 8bit Color Depth Color or more

8 bits * 3 colors = 24 bit for True Color
vs 30 bits * 3 colors = 30 bit color.
2^2 = 4
2^3 = 8
2^4 = 16
2^5 = 32
2^6 = 64
2^7 = 128
2^8 = 256
Color of Human Eye vs TV
- Human Eye   = 10 million colors. 
- Human Vision Color Depth = 8bits per color. We see 3 such colors, Red, Green & Blue. *We can see 256 shades of a color.*
- TV          = 24 bits for rgb = 2^24 = 16.7 million colors
- TV          = 30 bits for rgb = 2^30 = 1.07 billion colors = 64 times more colors than 24 bit

vs "Total Pixels with that Color Depth"
- Human Eye       = Total Simulated Megapixels 576. Actual megpixels = 7 mega pixels at fovea
- TV Resolution   = 1920 * 1080 pixels (1080p)    = 2    mega pixels 
- TV Resolution   = 3840 * 2160 pixels (4K)       = 8.3  mega pixels
- TV Resolution   = 7680 * 4320 pixels (8K)       = 16.3 mega pixels