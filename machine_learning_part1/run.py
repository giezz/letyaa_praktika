from scipy.io import loadmat
import numpy as np
from displayData import displayData
from predict import predict
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = loadmat('test_set.mat')
    y = data['y']
    X = data['X']

    data = loadmat('weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']
    m = len(X)

    randomPermutation = np.random.permutation(m)
    examples = []
    for x in range(0, 100, 1):
        examples.append(X[randomPermutation[x]])
    displayData(np.array(examples), '100 random samples')

    pred = predict(Theta1, Theta2, X)

    y_vector = y.ravel()
    matchedValues = 0
    unmatchedValues = []
    for i in range(0, 5000, 1):
        if y_vector[i] == pred[i]:
            matchedValues += 1
        else:
            unmatchedValues.append(i)
    print(f'The average number of matched values is {np.double(matchedValues / 5000) * 100} %')

    rp = np.random.permutation(m)
    plt.figure()
    for i in range(5):
        X2 = X[rp[i], :]
        X2 = np.matrix(X[rp[i]])
        pred = predict(Theta1, Theta2, X2.getA())
        pred = np.squeeze(pred)
        pred_str = 'Neural network assumption ' + str(y_vector[rp[i]])
        displayData(X2, pred_str)
    plt.close()

    Examples = []
    rp = np.random.permutation(m)
    print('Total incorrect assumptions ', str(len(unmatchedValues)))
    for x in range(0, 100, 1):
        Examples.append(X[rp[x]])
    displayData(np.array(Examples), '100 neural network fails')
