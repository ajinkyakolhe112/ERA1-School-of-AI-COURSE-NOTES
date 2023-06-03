#!/usr/bin/env python3

from torchinfo import summary


summary(model, input_size=(1, 28, 28), verbose=2,
		col_names=["input_size", "output_size", "kernel_size", "mult_adds", "num_params", "params_percent"], col_width=20);