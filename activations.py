import numpy as np


class Tanh:
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return (1 - self.output ** 2) * output_gradient


class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient
