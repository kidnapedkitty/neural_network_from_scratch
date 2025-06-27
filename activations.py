import numpy as np


class Tanh:
    def forward(self, input, training=True):
        self.output = np.tanh(input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        return (1 - self.output ** 2) * output_gradient


class ReLU:
    def forward(self, input, training=True):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        input_gradient[self.input <= 0] = 0
        return input_gradient


class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            self.mask = np.random.binomial(1, 1.0 - self.dropout_rate, size=input_data.shape)

            return (input_data * self.mask) / (1.0 - self.dropout_rate)
        else:

            return input_data

    def backward(self, output_gradient, learning_rate):

        return output_gradient * self.mask
