import numpy as np


def add_zero_feature(X, axis=1):
    return np.append(np.ones((X.shape[0], 1) if axis else (1, X.shape[1])), X, axis=axis)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
