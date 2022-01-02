import numpy as np


def MSE(labels, outputs):
    # mean squared error loss
    return np.mean(np.power(labels - outputs, 2))


def MSE_prime(labels, outputs):
    # derivative of MSE
    return 2 * (outputs - labels)/np.size(labels)


def binary_crossentropy(labels, outputs):
    outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)  # avoids div by 0
    return -np.mean((1 - labels) * np.log(1 - outputs_clipped) + labels * np.log(outputs_clipped))


def binary_crossentropy_prime(labels, outputs):
    # TODO check this
    outputs_clipped = np.clip(outputs, 1e-15, 1-1e-15)  # avoids div by 0
    return np.mean(((1-labels)/(1-outputs_clipped) - labels/outputs_clipped))

# TODO Implement
def multiclass_crossentrpy(labels, outputs): 
    return 0
    
# TODO Implement
def multiclass_crossentrpy_prime(labels, outputs): 
    return 0

def MEE(label1, label2, output1, output2):
    # mean euclidean error loss
    return np.mean(np.sqrt(np.power(label1-output1, 2)+np.power(label2-output2, 2)))

#TODO Implement
def MEE_prime(label1, label2, output1, output2):
    return 0
