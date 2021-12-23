import numpy as np

def L2(weights, l=0.005): #L2 norm regularization
    norm = np.linalg.norm(weights)
    return l*norm

def weight_decay(weights, l): #weight-decay regularization
    return l*weights
