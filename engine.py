def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss_function, loss_derivative, X_train, y_train, epochs=10000, learning_rate=0.01, verbose=True):
    for epoch in range(epochs):
        error = 0
        for x, y in zip(X_train, y_train):
            # forward pass
            output = predict(network, x)

            # error calculation
            error += loss_function(y, output)

            # backward pass
            output_gradient = loss_derivative(y, output)
            for layer in reversed(network):
                output_gradient = layer.backward(output_gradient, learning_rate)

        error /= len(X_train)

        if verbose:
            print(f'Epoch {epoch + 1}/{epochs}, Error: {error:.4f}')
