from random import random
from activationfunctions import LeakyReLU, ReLU, Sigmoid, Tanh
from losses import MEE
from layers import FullyConnectedLayer
from metrics import MeanEuclideanError
from neuralnetwork import Network
from utils import plot_and_save, tr_vl_split, compare, multiline_plot
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
    net = Network("CUP", MEE(), nesterov=0.25)
    net.add(FullyConnectedLayer(10, 23, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(23, 2, initialization_func="normalized_xavier"))
    history = net.training_loop(Xtr, Ytr, epochs=1549, learning_rate=0.025, verbose=True, batch_size=len(Xtr))

    # Model evaluation
    err = MeanEuclideanError().compute(net, Xts, Yts)
    return err, net, history


# newnet = Network("Current Best")
# newnet.loadnet("models/CUP_currentbest.pkl")
# newnet.summary()
# # Model evaluation
# Xts, Yts = load_cup(file="test")
# err = MeanEuclideanError().compute(newnet, Xts, Yts)
# print("Current best: ", err)
print("Assessing given network on CUP" )
vals = []
hists, tags, epochs = [], [], []
for i in range(50):
    suffix = "CUP_nesterov/CUP_nesterovbest_eval" + str(i)
    err, net, hist = CUP_evaluation()
    print("MEE:", err)
    vals.append(err)
    hists.append(hist)
    tags.append(str(i))
    epochs.append(len(hist) - 50)

net.summary()
net.savenet("models/CUP_nesterovbest.pkl")
multiline_plot("CUP Evaluation (internal TS)", hists, tags, xlabel="Epochs", ylabel="Loss", style="Spectral", savefile="CUP_nesterov/CUP_eval_history", showlegend=False)
print("MEE", np.mean(vals), "+-", np.std(vals))
print("Epochs", np.mean(epochs), "+-", np.std(epochs))

"""
+==== Structure
|IN     FullyConnectedLayer: 10 units with Tanh activation function
|HID    FullyConnectedLayer: 23 units with Tanh activation function
|OUT    FullyConnectedLayer: 2 units
+==== Loss: Mean Euclidean Error and momentum = 0.25 and Nesterov momentum
For a total of 301 trainable parameters
MEE 0.18739041656939193 +- 0.008833923071037302
Epochs 1549
Learning rate = 0.00125
Batch size = 16
"""