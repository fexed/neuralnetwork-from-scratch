import numpy as np


class Loss():
    """ Base class for the loss functions """

    def __init__(self):
        self.name = None


class MEE(Loss):
    """ Mean Euclidean Error loss """

    def __init__(self):
        self.name = "Mean Euclidean Error"


    def compute(self, outputs, targets):  # mean euclidean error loss
        return np.mean(np.sqrt(np.sum(np.square(targets - outputs))))


    def derivative(self, outputs, targets):  # derivative of MEE
        return (outputs - targets)/(np.sqrt(np.sum(np.square(targets - outputs)))*np.size(targets))


class MSE(Loss):
    """ Mean Squared Error loss """

    def __init__(self):
        self.name = "Mean Squared Error"


    def compute(self, outputs, targets):  # mean squared error loss
        return np.mean(np.power(targets - outputs, 2))


    def derivative(self, outputs, targets):  # derivative of MSE
        return 2 * (targets - outputs)/np.size(targets)


class BinaryCrossentropy(Loss):
    """ Binary Crossentropy loss """

    def __init__(self):
        self.name = "Binary Crossentropy"


    def compute(self, outputs, targets):
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)  # avoids div by 0
        return np.mean(-(1 - targets) * np.log(1 - outputs_clipped) - targets * np.log(outputs_clipped))


    def derivative(self, outputs, targets):  # TODO check this
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)  # avoids div by 0
        return np.mean(targets/outputs_clipped - (1-targets)/(1-outputs_clipped))


class MulticlassCrossentropy(Loss):
    """ Multiclass Crossentropy loss """

    # TODO implement
    def __init__(self):
        self.name = "Multiclass Crossentropy"


    def compute(self, outputs, targets):
        return 0


    def derivative(self, outputs, targets):
        return 0
