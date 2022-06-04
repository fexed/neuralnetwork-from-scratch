import numpy as np


class Regularizator():
        """ Base class for the regularation functions """

        def __init__(self, l):
            self.name = None
            self.l = 0


class L1(Regularizator):
        """ L1 or Lasso regularizator """

        def __init__(self, l = 0.005):
            self.name = "L1"
            self.l = l


        def forward(self, weights):
            return self.l * np.sum(np.abs(weights))


        def derivative(self, weights):
            out = []
            for l1 in weights:
                outt = []
                for l2 in l1:
                    if l2 < 0:  outt.append(-self.l)
                    elif l2 > 0: outt.append(self.l)
                    else: outt.append(0)
                out.append(outt)
            return out


class L2(Regularizator):
        """ L2 or Tikhonov regularizator """

        def __init__(self, l = 0.005):
            self.name = "L2"
            self.l = l


        def forward(self, weights):
            return self.l * np.sum(np.square(weights))


        def derivative(self, weights):
            return 2 * self.l * weights


class Thrun(Regularizator):
        """ Regularizator from Monk paper """

        def __init__(self, l = 0.005):
            self.name = "Thrun"
            self.l = l

        def forward(self, weights):
            return self.l * np.sum(np.power(weights, 4))/4 +  self.l * np.sum(np.square(weights))

        def derivative(self, weights):
            return self.l * np.power(weights, 3) + 2 * self.l * weights