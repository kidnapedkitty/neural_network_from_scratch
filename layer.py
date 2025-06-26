class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # takes input and returns output
        pass

    def backward(self, output_gradient, learning_rate):
        # takes derivative of loss with respect to output and returns derivative of loss with respect to input, also updates weights
        pass
