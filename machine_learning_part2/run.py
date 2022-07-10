# coding=utf-8
from scipy.io import loadmat
import numpy as np
from scipy.optimize import minimize
from functions import add_zero_feature, sigmoid


def transform_arguments(transformation):
    def dec(f):
        def wrapper(*args, **kwargs):
            t_args = map(transformation, args)
            t_kwargs = {k: transformation(v) for k, v in kwargs.iteritems()}
            return f(*t_args, **t_kwargs)

        return wrapper

    return dec


matrix_args_array_only = transform_arguments(
    lambda arg: np.matrix(arg, copy=False) if isinstance(arg, np.ndarray) else arg)


@matrix_args_array_only
def cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    Theta1 = nn_params[0, :hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = nn_params[0, hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))

    m = Y.shape[1]
    Y = Y.A

    A_1 = X
    Z_2 = Theta1 * A_1.T
    A_2 = sigmoid(Z_2)
    A_2 = add_zero_feature(A_2, axis=0)
    Z_3 = Theta2 * A_2
    A_3 = sigmoid(Z_3)
    H = A_3.A

    J = np.sum(-Y * np.log(H) - (1 - Y) * np.log(1 - H)) / m

    reg_J = 0.0
    reg_J += np.sum(np.power(Theta1, 2)[:, 1:])
    reg_J += np.sum(np.power(Theta2, 2)[:, 1:])

    J += reg_J * (float(lambda_coef) / (2 * m))

    return J


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), 1 - sigmoid(z))


def rand_initialize_weights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init


def predict(Theta1, Theta2, X):
    h1 = sigmoid(np.dot(X, Theta1.T))
    h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))
    y_pred = np.argmax(h2, axis=1) + 1
    return y_pred



@matrix_args_array_only
def gradient_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef):
    Theta1 = nn_params[0, :hidden_layer_size * (input_layer_size + 1)].reshape(
        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = nn_params[0, hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))

    m = Y.shape[1]

    A_1 = X
    Z_2 = Theta1 * A_1.T
    A_2 = sigmoid(Z_2)
    A_2 = add_zero_feature(A_2, axis=0)
    Z_3 = Theta2 * A_2
    A_3 = sigmoid(Z_3)

    DELTA_3 = A_3 - Y
    DELTA_2 = np.multiply((Theta2.T * DELTA_3)[1:, :], sigmoid_gradient(Z_2))
    Theta1_grad = (DELTA_2 * A_1) / m
    Theta2_grad = (DELTA_3 * A_2.T) / m

    lambda_coef = float(lambda_coef)
    Theta1_grad[:, 1:] += (lambda_coef / m) * Theta1[:, 1:]
    Theta2_grad[:, 1:] += (lambda_coef / m) * Theta2[:, 1:]

    return np.concatenate((Theta1_grad.A1, Theta2_grad.A1))


if __name__ == '__main__':
    # Задание 1.
    # Загрузить обучающую выборку из файла training_set.mat в переменные X и y
    # Загрузить весовые коэффициенты из файла weights.mat в переменные Theta1 и Theta2
    # Использовать для этого функцию scipy.io.loadmat

    # Задание 2.
    # Программно определить параметры нейронной сети
    # input_layer_size = ...  # количество входов сети (20*20=400)
    # hidden_layer_size = ... # нейронов в скрытом слое (25)
    # num_labels = ...        # число распознаваемых классов (10)
    # m = ...                 # количество примеров (5000)

    data = loadmat('test_set.mat')
    y = data['y']
    X = data['X']

    data = loadmat('weights.mat')
    Theta1 = data['Theta1']
    Theta2 = data['Theta2']

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    m = len(y)

    # добавление единичного столбца - нейрон смещения
    X = add_zero_feature(X)

    # декодирование вектора Y
    # Y = decode_y(y)
    Y = (np.arange(num_labels)[:, np.newaxis] == (y.T - 1)).astype(float)

    # объединение матриц Theta в один большой массив
    # nn_params = pack_params(Theta1, Theta2)
    nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()))

    # проверка функции стоимости для разных lambda
    lambda_coef = 0
    print('Функция стоимости для lambda {} = {}'.
          format(lambda_coef, cost_function(
        nn_params, input_layer_size, hidden_layer_size,
        num_labels, X, Y, lambda_coef)))

    lambda_coef = 1
    print('Функция стоимости для lambda {} = {}'.
          format(lambda_coef, cost_function(
        nn_params, input_layer_size, hidden_layer_size,
        num_labels, X, Y, lambda_coef)))

    # проверка производной sigmoid
    gradient = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Производная функции sigmoid в точках -1, -0.5, 0, 0.5, 1:')
    print(gradient)

    # случайная инициализация параметров
    initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    initial_nn_params = np.concatenate((initial_Theta1.ravel(), initial_Theta2.ravel()))

    # обучение нейронной сети
    res = minimize(cost_function, initial_nn_params, method='L-BFGS-B',
                   jac=gradient_function, options={'maxiter': 100},
                   args=(input_layer_size, hidden_layer_size, num_labels, X, Y, lambda_coef)).x

    # разбор вычисленных параметров на матрицы Theta1 и Theta2
    Theta1 = res[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, (input_layer_size + 1)))
    Theta2 = res[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, (hidden_layer_size + 1)))

    # выичисление отклика сети на примеры из обучающей выборки
    # h1 = sigmoid(np.dot(X, Theta1.T))
    # h2 = sigmoid(np.dot(add_zero_feature(h1), Theta2.T))
    # y_pred = np.argmax(h2, axis=1) + 1

    print('Точность нейронной сети на обучающей выборке: {}'.format(
        np.mean(predict(Theta1, Theta2, X) == y.ravel(), ) * 100))
