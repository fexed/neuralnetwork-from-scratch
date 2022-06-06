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
    net = Network("CUP", MEE(), regularizator=L2(l=0.0001))
    net.add(FullyConnectedLayer(10, 21, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(21, 2, initialization_func="normalized_xavier"))
    history = net.training_loop(Xtr, Ytr, epochs=200, learning_rate=0.00125, verbose=True, batch_size=1)

    # Model evaluation
    err = MeanEuclideanError().compute(net, Xts, Yts)
    return err, net, history


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
    epochs.append(len(hist))

net.summary()
net.savenet("models/CUP_nesterovbest.pkl")
multiline_plot("CUP Evaluation (internal TS)", hists, tags, xlabel="Epochs", ylabel="Loss", style="Spectral", savefile="CUP_nesterov/CUP_eval_history", showlegend=False)
print("MEE", np.mean(vals), "+-", np.std(vals))
print("Epochs", np.mean(epochs), "+-", np.std(epochs))

"""
+==== Structure
|IN     FullyConnectedLayer: 10 units with Tanh activation function
|HID    FullyConnectedLayer: 21 units with Tanh activation function
|OUT    FullyConnectedLayer: 2 units
+==== Loss: Mean Euclidean Error and L2 regularizator with lambda = 0.0001
For a total of 506 trainable parameters
MEE 0.1499942430775244 +- 0.004481060356927671
Epochs 200
Learning rate = 0.00125
Batch size = 1
"""