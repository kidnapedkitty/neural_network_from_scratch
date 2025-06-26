import numpy as np


def mse_loss(predicted, actual):
    return np.mean(np.power(actual - predicted, 2))


def mse_loss_derivative(predicted, actual):
    return 2 * (predicted - actual) / np.size(actual)
