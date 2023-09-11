
# Revision but Constructive. From systems design perspective. Entire end to end

## Essential Pytorch

### Data Type
Single High Level Data Type: `torch.Tensor`. 
Python has lists, dictionary, variables, string, Pytorch has just one Tensor.
Imp `Tensor` methods
1. `<tensor>.to`
1. `<tensor>.reshape`
1. `<tensor>.shape`

### Data Type Simple Methods
- `torch` methods

### Data Type, complex Methods
`torch.nn` methods
1. `nn.Conv2d` 
2. `nn.Linear`
3. `nn.Sequential`

Python Hidden Features & their Cost - [[Principals of Programming Languages]]
- Dynamic Memory Creation
- Dynamic Data Type. (Cost: Doesn't have type checking)
- Interpreted Language. (Price: Slow. Advantage: Edit, Test, Debug)
- Design Level: High Level
- Basic Libraries but well Designed (Cost: Need specialized libraries for specialized things)
- Dynamic Functions, Can't declare functions. Because its a interpreted language
(Beware of Shortcuts to Knowledge. It will surely extact it's price.) 

----

[[Library API design]]
- Pytorch design
- Gluon design
- Scikit Learn design
- Keras design

[[Computer -> Calculator -> Machine. Engineering]]
- Computer as Advanced Calculator Machine
- Human Body as Machine
- Car as Machine

[[Computer Architecture]]
SIMD - CPU or GPU (many SIMD).. (Single Instruction and Multiple Data. Makes vectorization possible. Add(X) )
- Operation = CPU
- Data = Register
- `BUT Register <- RAM <- Large Storage`. Two bottlenecks. RAM = storage of Model. And RAM storing batch for execution


