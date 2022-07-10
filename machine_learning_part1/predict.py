import numpy as np


def predict(Theta1, Theta2, X):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def add_zero_feature(X, axis=1):
        return np.append(np.ones((X.shape[0], 1) if axis else (1, X.shape[1])), X, axis=axis)

    X = add_zero_feature(X)
    h1 = sigmoid(np.dot(X, Theta1.T))
    h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))
    y_pred = np.argmax(h2, axis=1) + 1
    return y_pred
