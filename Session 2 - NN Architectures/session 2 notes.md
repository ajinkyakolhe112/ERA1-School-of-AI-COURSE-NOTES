
## Imp Architecture Details
1. Convolution in Detail
   1. Learning AID for it?
2. Stride, Padding
3. Receptive Field Formalue, N_out formulae calculations (Excel, Markdown Latex, Code Latex Print)
4. GPU Parallelization of Convolution Operations & Other operations. Pytorch
5. Pytorch model Sequential -> nn.Module & Functional
6. Data Loader for model

## NN Design PLACES SELF
1. *FIRST**Tensorflow Playground Visual Architecture
1. **SECOND**. Use EXCEL. Visual formula. Always
2. **THIRD**Use Markdown Maths Equation. Maths Formula
3. Code: Keras just for architecture
4. Code: Pytorch
6. Text File, Numerical Calculation
7. Markdown Table


Advanced Question.
1. Why doesn't pytorch have reshape module. It would make things much easier. 

explicit choice of library owners. https://github.com/pytorch/pytorch/issues/2486#issuecomment-326121073
nn.Sequential() was also relunctantly provided. Because its easy, it also encourages that habit. Easier way becomes the default, and more powerful and customizable doesn't get used at all. 

### Library Performance
Library performance is important. & Impact on it from philosophy choices
Philosophy of Clean Coding
Philosophy of Python
Philosophy of Pytorch
1. Designed for easy pythonic reading
2. But also constraints of customizable, accelerated with good compiler, and accelerated with good hardware. 
3. feature rich library comes at cost of execution speed and memory footprint. 

In DL every parameter, every pixel value and its data type, its data storage volume, its read and write access speed, its parallelizability, flop speed are all important. 
that's why we want a library with essential , well optmized and well designed features with code readable for humans and high performance.. trade of was done in number of operations to support.