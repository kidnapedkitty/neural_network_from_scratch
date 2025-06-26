import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)

        self.bias = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights.T, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weight_gradient = np.dot(self.input, output_gradient.T)

        input_gradient = np.dot(self.weights, output_gradient)

        self.weights -= learning_rate * weight_gradient
        self.bias -= learning_rate * output_gradient

        return input_gradient
