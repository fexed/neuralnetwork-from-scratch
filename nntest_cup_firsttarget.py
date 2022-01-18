from activationfunctions import sigmoid, sigmoid_prime
from losses import MEE, MEE_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from kfold import KFold
from preprocessing import continuous_standardizer
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import plot_loss, tr_vl_split
import numpy as np
import matplotlib.pyplot as plot
import time
import pickle


def compare(a, b, tollerance=1e-03):
    return abs(a - b) <= tollerance * max(abs(a), abs(b))


def test_CUP(output=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    if (output): print("\n\n****CUP")
    X, Y = load_cup(verbose=output, test=False)
    Y = Y[0:,0:,1]  # first target
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio=0.1)
    suffix = "CUP_" + ts
    net = Network("CUP test", MEE, MEE_prime)
    net.add(FullyConnectedLayer(10, 15, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(15, 15, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(15, 1, initialization_func="normalized_xavier"))
    if (output): net.summary()
    history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.001, verbose=output, early_stopping=25, batch_size=1)

    # accuracy on validation set
    out = net.predict(xvl)
    accuracy = 0
    for i in range(len(out)):
        # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
        if (compare(yvl[i][0], out[i][0])): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100

    if (output): print("Accuracy: {:.4f}%".format(accuracy))

    plot_loss(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
    return accuracy

acc = []
for i in range(100):
    acc.append(test_CUP(output=True))
print("Media: " + "{:.5f}%".format(np.mean(acc)))
print("Dev: " + "{:.5f}%".format(np.std(acc)))
