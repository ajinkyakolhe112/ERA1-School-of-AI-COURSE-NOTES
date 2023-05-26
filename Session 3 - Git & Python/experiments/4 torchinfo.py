#!/usr/bin/env python3

!pip install torchinfo # has more advanced functionality
from torchinfo import summary

summary(model,input_size=(1,28,28),verbose=2,
	col_names=["input_size","kernel_size", "output_size", "num_params", "params_percent"],col_width=20);