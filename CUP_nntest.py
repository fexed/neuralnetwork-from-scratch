from activationfunctions import Tanh
from losses import MEE
from layers import FullyConnectedLayer
from neuralnetwork import Network
from kfold import KFold
from preprocessing import continuous_standardizer
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import plot_and_save, tr_vl_split, compare
from metrics import Accuracy
import numpy as np
import matplotlib.pyplot as plot
import time
import pickle


def test_CUP(output=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    if (output): print("\n\n****CUP")
    X, Y = load_cup(verbose=output, file="training")
    # train
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio = 0.5)
    suffix = "CUP_" + ts
    net = Network("CUP test", MEE(), momentum=0.85)
    net.add(FullyConnectedLayer(10, 30, Tanh(), initialization_func="he"))
    net.add(FullyConnectedLayer(30, 2, initialization_func="he"))
    if (output): net.summary()
    history, val_history, accuracy_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.001, verbose=output, early_stopping=25, batch_size=1, lr_decay=None, metric = Accuracy())

    # accuracy on validation set
    acc = Accuracy().compute(net, xvl, yvl)

    if (output): print("Accuracy: {:.4f}%".format(acc))

    plot_and_save(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
    plot_and_save(title=suffix, history=accuracy_history, ylabel="Accuracy", xlabel="Epochs", savefile=suffix + "_accuracy")

test_CUP(output=True)
