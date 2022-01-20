import numpy as np


class Regularizator():
        """ Base class for the regularation functions """

        def __init__(self, lambda):
            self.name = None
            self.lambda = 0


class L2(Regularizator):
        """ L2 regularizator """

        def __init__(self, l = 0.005):
            self.name = L2
            self.lambda = l

        def apply(self, weights):
            return 2 * self.lambda * weights
