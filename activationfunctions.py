import numpy as np


class ActivationFunction:
    """ Base class for the activation functions """

    def __init__(self):
        self.name = None


class Tanh(ActivationFunction):
    """ Tanh activation function """

    def __init__(self):
        self.name = "Tanh"


    def forward(self, x):
        """ Returns the tanh value of x """

        return np.tanh(x)


    def derivative(self, x):
        """ Derivative of the tanh function """

        return 1 - np.power(np.tanh(x), 2)


class Sigmoid(ActivationFunction):
    """ Sigmoid activation function """

    def __init__(self):
        self.name = "Sigmoid"


    def forward(self, x):
        """ Returns the sigmoid value of x """

        # 1/(1+e^{-x})
        return 1 / (1 + np.exp(np.dot(-1, x)))


    def derivative(self, x):
        """ Derivative of the sigmoid function """

        sigma = self.forward(x)
        return sigma * (1 - sigma)


class ReLU(ActivationFunction):
    """ ReLU activation function """

    def __init__(self):
        self.name = "ReLU"


    def forward(self, x,  almost_zero = 1e-5):  # returns the positive part of x
        """ Returns the positive part of x """
        return np.maximum(x, almost_zero)


    def derivative(self, x, almost_zero = 1e-5 ):  # derivative of the ReLU function
        """ Derivative of the ReLU function """
        return np.array(1 if x > 0 else almost_zero )


class LeakyReLU(ActivationFunction):
    """ Leaky ReLU activation function """

    def __init__(self, leak = 0.3):
        self.name = "ReLU"
        self.leak = leak


    def forward(self, x):
        """ Returns the positive part of x or allows a
        small gradient if x <= 0
        """

        return ((x > 0) * x) + ((x <= 0) * self.leak * x)


    def derivative(self, x):
        """ Derivative of the leaky ReLU function """

        return (x > 0) + ((x <= 0) * self.leak)


class Softmax(ActivationFunction):  # TODO: check this
    """ Softmax activation function """

    def __init__(self):
        self.name = "Softmax"


    def forward(self, x):
        """ Returns the softmax value of x """

        exp = np.exp(x - x.max())  # subtract max for numerical stability
        return exp / exp.sum(axis = 0)


    def derivative(self, x):
        """ Derivative of the softmax function """
        
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
