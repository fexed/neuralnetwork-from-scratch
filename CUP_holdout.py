from random import random
from activationfunctions import LeakyReLU, ReLU, Sigmoid, Tanh
from losses import MEE
from layers import FullyConnectedLayer
from metrics import MeanEuclideanError
from neuralnetwork import Network
from utils import plot_and_save, tr_vl_split, compare
from regularizators import L2
import numpy as np
import matplotlib.pyplot as plot
from preprocessing import continuous_standardizer, min_max_normalizer
from dataset_loader import load_cup


def CUP_evaluation():
    # Training set loading
    X, Y = load_cup()

    # Preprocessing
    X, n_min, n_max = min_max_normalizer(X)
    X, means, std = continuous_standardizer(X)

    X_TR,  X_VAL, Y_TR, Y_VAL = tr_vl_split(X, Y, ratio=0.2)

    # Training
    net = Network("CUP", MEE(), regularizator=L2(l = 1e-06))
    net.add(FullyConnectedLayer(10, 24, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(24, 2,  initialization_func="normalized_xavier"))
    history = net.training_loop(X_TR, Y_TR, epochs=1300, learning_rate=0.000625, verbose=False, batch_size=1)

    # Model evaluation
    err = MeanEuclideanError().compute(net, X_VAL, Y_VAL)
    return err, net


print("Assessing given network on CUP" )  # NEED INTERNAL TEST SET
vals = []
for i in range(10):
    err, net = CUP_evaluation()
    vals.append(err)

net.summary()
print("MEE", np.mean(vals), "+-", np.std(vals))