import numpy as np
from layer import Layer


class Activation(Layer):
    def __init__(self, activation_function, derivative_function):
        self.activation_function = activation_function
        self.derivative_function = derivative_function

    def forward(self, input):
        self.input = input
        return self.activation_function(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.derivative_function(self.input))
