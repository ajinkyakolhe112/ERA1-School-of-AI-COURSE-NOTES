1. Convolution in Detail
   1. Learning AID for it?
2. Stride, Padding
3. Receptive Field Formalue, N_out formulae calculations (Excel, Latex)
4. GPU Parallelization of Convolution Operations & Other operations. Pytorch
5. Pytorch model Sequential -> nn.Module & Functional
6. Data Loader for model



Advanced Question.
Why doesn't pytorch have reshape module. It would make things much easier. 
explicit choice of library owners. https://github.com/pytorch/pytorch/issues/2486#issuecomment-326121073
nn.Sequential() was also relunctantly provided. Because its easy, it also encourages habits. Easier way becomes the default, and more powerful and customizable doesn't get used at all. 


Library performance is important. 
Philosophy of Clean Coding
Philosophy of Python
Philosophy of Pytorch
1. Designed for easy pythonic reading
2. But also constraints of customizable, accelerated with good compiler, and accelerated with good hardware. 
3. feature rich library comes at cost of execution speed and memory footprint. 

In DL every parameter, every pixel value and its data type, its data storage volume, its read and write access speed, its parallelizability, flop speed are all important. 
that's why we want a library with essential , well optmized and well designed features with code readable for humans and high performance.. trade of was done in number of operations to support.