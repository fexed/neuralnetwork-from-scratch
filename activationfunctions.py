import numpy as np


def tanh(x):
    return np.tanh(x);


def tanh_prime(x):
    return 1 - np.power(np.tanh(x), 2);


def sigmoid(x):
    return 1 / (1 + np.exp(np.dot(-1, x)))  # 1/(1+e^{-x})


def sigmoid_prime(x):
    sigma = sigmoid(x)
    return sigma * (1 - sigma)


def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x > 0).astype('int')


def leaky_relu(x, leak=0.3):
    return ((x > 0) * x) + ((x <= 0) * leak * x)


def leaky_relu_prime(x, leak=0.3):
    return (x > 0) + ((x <= 0) * leak)


def softmax(x):
    exp = np.exp(x - x.max())  # subtract max for numerical stability
    return exp / exp.sum(axis = 0)


def softmax_deriv(x):
    jacobian_m = np.diag(x)
    s = []

    for i in range(len(jacobian_m)):
        s.append([])
        for j in range(len(jacobian_m)):
            if i == j:
                s[i].append(x[i] * (1-x[i]))
            else:
                s[i].append(- x[i] * x[j])
    return np.array(s)
