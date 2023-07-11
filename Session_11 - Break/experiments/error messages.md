```python
Traceback (most recent call last):
  File "./experiments/model_commentary.py", line 62, in <module>
    model(test_img)
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "./experiments/model_commentary.py", line 53, in forward
    x = self.features(x)
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/container.py", line 217, in forward
    input = module(input)
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/nn/modules/pooling.py", line 166, in forward
    return F.max_pool2d(input, self.kernel_size, self.stride,
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/_jit_internal.py", line 484, in fn
    return if_false(*args, **kwargs)
  File "/Users/ajinkya/opt/anaconda3/lib/python3.9/site-packages/torch/nn/functional.py", line 782, in _max_pool2d
    return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
RuntimeError: Given input size: (192x2x2). Calculated output size: (192x0x0). Output size is too small
```