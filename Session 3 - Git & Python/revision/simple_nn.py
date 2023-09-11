import torch

architecture = [784, 5, 2]

x_examples    = torch.randn(  size= (1000,784),   dtype= torch.float32,   requires_grad= False    )
y_labels      = torch.randn(  size= (1000),       dtype=torch.flaot32,    requires_grad= False    )

class Neuron:
    neurons_num = 1
    connections = torch.randn(  size= (5, 784),     dtype= torch.float32,   requires_grad= True     )
    bias        = torch.ones_like(neurons_num)

    def forward_pass(inputs: torch.Tensor):
        weighted_sum = torch.dot(inputs, connections) + bias
        activation   = activation_function(weighted_sum)
        return activation
    
    def activation_function(weighted_sum: torch.Tensor):
        activation = torch.relu(weighted_sum)
        return activation

    def get_parameters(model: torch.nn.Sequential):
        parameters = torch.randn(size=())

        return parameters

    def get_parameter_gradients(model: torch.nn.Sequential):
        gradients = torch.randn(size=())
        
        return gradients

    def update_parameters(parameter_tensors, gradient_tensors):
        for parameter, gradient in zip(parameter_tensors, gradient_tensors):
            parameter = parameter - gradient * learning_rate
    
    def calculate_gradients(operations_with_parameters):
        y_output = function(examples, connections, weighted_sums, bias, activations)
        y_labels = y_labels
        error    = y_output - y_labels
        pass

"""
one to one mapping between `example <-> single neuron connections`
"""

