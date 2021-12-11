import numpy as np

def L2(weights, l=0.005):
    sum = 0
    for weight in weights:
        for w in weight:
            sum += l*w
    return sum
