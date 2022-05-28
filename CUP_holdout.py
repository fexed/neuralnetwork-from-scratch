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
    Xtr, Xvl, Ytr, Yvl = tr_vl_split(Xtr, Ytr, ratio=0.1)
    net = Network("CUP", MEE(), nesterov=True, momentum=0.25)
    net.add(FullyConnectedLayer(10, 23, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(23, 2, initialization_func="normalized_xavier"))
    history, val_history = net.training_loop(Xtr, Ytr, X_validation=Xvl, Y_validation=Yvl, epochs=4000, learning_rate=0.00125, verbose=True, batch_size=16, early_stopping=50)

    # Model evaluation
    err = MeanEuclideanError().compute(net, Xts, Yts)
    plot_and_save(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
    return err, net, history, val_history


newnet = Network("Current Best")
newnet.loadnet("models/CUP_currentbest.pkl")
newnet.summary()
# Model evaluation
Xts, Yts = load_cup(file="test")
err = MeanEuclideanError().compute(newnet, Xts, Yts)
print("Current best: ", err)
print("Assessing given network on CUP" )
vals = []
hists, val_hists, epochs = [], [], []
for i in range(10):
    suffix = "CUP_nesterov/CUP_nesterov.25_1L_23U_eval_" + str(i)
    err, net, hist, val_hist = CUP_evaluation()
    print("MEE:", err)
    vals.append(err)
    hists.append(hist)
    val_hists.append(val_hist)
    epochs.append(len(hist) - 50)

net.summary()
#net.savenet("models/CUP_nesterovbest.pkl")
#plot_and_save(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
print("MEE", np.mean(vals), "+-", np.std(vals))
print("Epochs", np.mean(epochs), "+-", np.std(epochs))

"""
+==== Structure
|IN     FullyConnectedLayer: 10 units with Tanh activation function
|HID    FullyConnectedLayer: 23 units with Tanh activation function
|OUT    FullyConnectedLayer: 2 units
+==== Loss: Mean Euclidean Error and momentum = 0.25 and Nesterov momentum
For a total of 301 trainable parameters
MEE 0.19218554503029459 +- 0.007820414363134251
Epochs 1290.9 +- 467.80582510268084
Learning rate = 0.00125
Batch size = 16
"""