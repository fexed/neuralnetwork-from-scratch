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


# best {'loss': 0.4360415384225904, 'layers': 2, 'units': 30, 'learning_rate': 0.001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0, 'regularizator': None, 'regularization_lambda': 0, 'dropout': 0, 'epochs': 291}
def test_CUP(output=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    if (output): print("\n\n****CUP")
    X, Y = load_cup(verbose=output, test=False)
    Y = Y[0:,0:,1]  # second target
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio=0.5)
    suffix = "CUP_" + ts
    net = Network("CUP test second target", MEE())
    net.add(FullyConnectedLayer(10, 30, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(30, 30, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(30, 30, Tanh(), initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(30, 1, initialization_func="normalized_xavier"))
    if (output): net.summary()
    history, val_history, accuracy_history, epochs_done = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=300, learning_rate=0.001, verbose=output, early_stopping=25, batch_size=1, metric = Accuracy())

    # accuracy on validation set
    accuracy = Accuracy().compute(net, xvl, yvl)

    if (output): print("Accuracy: {:.4f}%".format(accuracy))

    plot_and_save(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
    plot_and_save(title=suffix, history=accuracy_history, ylabel="Accuracy", xlabel="Epochs", savefile=suffix + "_accuracy")
    return accuracy

acc = []
for i in range(1):
    acc.append(test_CUP(output=True))
print("Media: " + "{:.5f}%".format(np.mean(acc)))
print("Dev: " + "{:.5f}%".format(np.std(acc)))
