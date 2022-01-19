import numpy as np


class ActivationFunction:
    def __init__(self):
        self.name = None


class Tanh(ActivationFunction):
    def __init__(self):
        self.name = "Tanh"


    def forward(self, x):  # returns the tanh value of x
        return np.tanh(x)


    def derivative(self, x):  # derivative of the tanh function
        return 1 - np.power(np.tanh(x), 2)


class Sigmoid(ActivationFunction):
    def __init__(self):
        self.name = "Sigmoid"


    def forward(self, x):  # returns the sigmoid value of x
        return 1 / (1 + np.exp(np.dot(-1, x)))  # 1/(1+e^{-x})


    def derivative(self, x):  # derivative of the sigmoid function
        sigma = self.forward(x)
        return sigma * (1 - sigma)


class ReLU(ActivationFunction):
    def __init__(self):
        self.name = "ReLU"


    def forward(self, x):  # returns the positive part of x
        return np.maximum(x, 0)


    def derivative(self, x):  # derivative of the ReLU function
        return np.array(x > 0).astype('int')


class ReLU(ActivationFunction):
    def __init__(self, leak = 0.3):
        self.name = "ReLU"
        self.leak = leak


    def forward(self, x):
        # returns the positive part of x or allows a small gradient if x <= 0
        return ((x > 0) * x) + ((x <= 0) * self.leak * x)


    def derivative(self, x):  # derivative of the leaky ReLU function
        return (x > 0) + ((x <= 0) * self.leak)


class Softmax(ActivationFunction):  # TODO: check this
    def __init__(self):
        self.name = "Softmax"


    def forward(self, x):  # return the softmax value of x
        exp = np.exp(x - x.max())  # subtract max for numerical stability
        return exp / exp.sum(axis = 0)


    def derivative(self, x):  # derivative of the softmax function
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
