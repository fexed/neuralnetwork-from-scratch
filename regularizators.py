import numpy as np
from hyperparameter import HyperParameter


class Regularizator(HyperParameter):
        """ Base class for the regularation functions """

        def __init__(self, type, l):
            super().__init__(name = type + "Regularization", training=False)
            self.key = 'regularizator'
            self.type = type
            self.l = l

        def value(self):
            return self

class L1(Regularizator):
        """ L1 or Lasso regularizator """ 

        def __init__(self, l = 0.005):
            super().__init__("L1", l)

        def forward(self, weights):
            return self.l * np.sum(np.abs(weights))


        def derivative(self, weights):
            return 2 * self.l * weights


class L2(Regularizator):
        """ L2 or Tikhonov regularizator """

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


class Thrun(Regularizator):
        """ Regularizator from Monk paper """

        def __init__(self, l = 0.005):
            super().__init__("Thrun", l)

        def forward(self, weights):
            return self.l * (np.sum(np.power(weights, 4)/4) +  np.sum(np.square(weights)))

        def derivative(self, weights):
            return self.l * (np.power(weights, 3) + 2 * self.l * weights)