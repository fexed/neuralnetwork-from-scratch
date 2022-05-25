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
    Xtr, Ytr = load_cup(file="training")
    Xts, Yts = load_cup(file="test")

    # Preprocessing
    Xtr, n_min, n_max = min_max_normalizer(Xtr)
    Xtr, means, std = continuous_standardizer(Xtr)

    # Training
    net = Network("CUP", MEE(), regularizator=L2(l = 1e-06), nesterov=True, momentum=0.8)
    net.add(FullyConnectedLayer(10, 24, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(24, 2,  initialization_func="normalized_xavier"))
    history = net.training_loop(Xtr, Ytr, epochs=1300, learning_rate=0.000625, verbose=True, batch_size=16)

    # Model evaluation
    Xts, n_min, n_max = min_max_normalizer(Xts)
    Xts, means, std = continuous_standardizer(Xts)
    err = MeanEuclideanError().compute(net, Xts, Yts)
    return err, net


print("Assessing given network on CUP" )
vals = []
for i in range(1):
    err, net = CUP_evaluation()
    vals.append(err)

net.summary()
print("MEE", np.mean(vals), "+-", np.std(vals))