import numpy as np

def L2(weights, l=0.005): #L2 norm regularization
    return l * np.sum(np.square(weights))
