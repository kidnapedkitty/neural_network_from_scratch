import numpy as np


def mse_loss(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
