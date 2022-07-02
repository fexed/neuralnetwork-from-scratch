import numpy as np


class Loss():
    """ Base class for the loss functions """

    def __init__(self):
        self.name = None


class MEE(Loss):
    """ Mean Euclidean Error loss """

    def __init__(self):
        self.name = "Mean Euclidean Error"


    def forward(self, labels, outputs):  # mean euclidean error loss
        return np.mean(np.sqrt(np.sum(np.square(labels - outputs))))


    def derivative(self, labels, outputs):  # derivative of MEE
        return (outputs - labels)/(np.sqrt(np.sum(np.square(labels - outputs)))*np.size(labels))


class MSE(Loss):
    """ Mean Squared Error loss """

    def __init__(self):
        self.name = "Mean Squared Error"


    def forward(self, labels, outputs):  # mean squared error loss
        return np.mean(np.power(labels - outputs, 2))


    def derivative(self, labels, outputs):  # derivative of MSE
        return 2 * (labels - outputs)/np.size(labels)


class BinaryCrossentropy(Loss):
    """ Binary Crossentropy loss """

    def __init__(self):
        self.name = "Binary Crossentropy"


    def forward(self, labels, outputs):
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)  # avoids div by 0
        return np.mean(-(1 - labels) * np.log(1 - outputs_clipped) - labels * np.log(outputs_clipped))


    def derivative(self, labels, outputs):  # TODO check this
        outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)  # avoids div by 0
        return np.mean(labels/outputs_clipped - (1-labels)/(1-outputs_clipped))


class MulticlassCrossentropy(Loss):
    """ Multiclass Crossentropy loss """

    # TODO implement
    def __init__(self):
        self.name = "Multiclass Crossentropy"


    def forward(self, labels, outputs):
        return 0


    def derivative(self, labels, outputs):
        return 0
