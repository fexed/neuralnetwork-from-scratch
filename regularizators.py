from turtle import forward
import numpy as np
from hyperparameter import HyperParameter


class Regularization(HyperParameter):
        """ Base class for the regularation functions """

        def __init__(self, type, l):
            super().__init__(name = type + "Regularization", training=False)
            self.key = 'regularization'
            self.type = type
            self.l = l

        def forward(self, _): 
            return 0

        def forward(self, _): 
            return 0

        def value(self):
            return self

        def __str__(self): 
            return f"{self.name} with lambda equal to {self.l}"


class L1(Regularization):
        """ L1 or Lasso regularization """ 

        def __init__(self, l = 0.005):
            super().__init__("L1", l)

        def forward(self, weights):
            return self.l * np.sum(np.abs(weights))


        def derivative(self, weights):
            return 2 * self.l * weights


class L2(Regularization):
        """ L2 or Tikhonov regularization """

        def __init__(self, l = 0.005):
            super().__init__("L2", l)

        def forward(self, weights):
            return self.l * np.sum(np.square(weights))

        def derivative(self, weights):
            out = []
            for l1 in weights:
                outt = []
                for l2 in l1:
                    outt.append(2 * self.l * l2)
                out.append(outt)
            return np.array(out)


class Thrun(Regularization):
        """ Regularization from Monk paper """

        def __init__(self, l = 0.005):
            super().__init__("Thrun", l)

        def forward(self, weights):
            return self.l * (np.sum(np.power(weights, 4)/4) +  np.sum(np.square(weights)))

        def derivative(self, weights):
            return self.l * (np.power(weights, 3) + 2 * self.l * weights)