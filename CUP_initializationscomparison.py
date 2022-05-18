import numpy as np
from activationfunctions import Tanh
from losses import MEE
from layers import FullyConnectedLayer
from metrics import MeanEuclideanError
from neuralnetwork import Network
from utils import tr_vl_split
from dataset_loader import load_cup
from regularizators import L2
import matplotlib.pyplot as plot
import time
import pickle
from itertools import zip_longest


def test_CUP(output=True, initialization="normalized_xavier"):
    if (output): print("\n\n****CUP")
    X, Y = load_cup()
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio=0.2)
    if (output): print("Training set of " + str(X.size) + " elements")
    net = Network("CUP test", MEE(), regularizator=L2(l = 1e-05))
    net.add(FullyConnectedLayer(10, 23, Tanh(), initialization_func=initialization))
    net.add(FullyConnectedLayer(23, 2,  initialization_func=initialization))
    # train
    if (output): net.summary()
    history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1500, learning_rate=0.0025, verbose=output, early_stopping=50)

    # MEE on validation set
    mee = MeanEuclideanError().compute(net, xvl, yvl)
    if (output): print("\n\nMEE on CUP validation set of {:.6f}%".format(MEE) + " over " + str(len(out)) + " elements")

    return mee, history, val_history



for initialization in ["xavier", "normalized_xavier", "he", "basic"]:
    print("Beginning tests on CUP with " + initialization)
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    acc, hist, val_hist, epochs = [], [], [], []
    for j in range(0, 100):
        mee, history, val_history = test_CUP(output=False, initialization=initialization)
        acc.append(mee)
        hist.append(history)
        val_hist.append(val_history)
        epochs.append(len(history))
        print(str(j+1), end=" ", flush=True)
    print("")
    print("CUP with " + initialization)
    mean = np.mean(acc)
    std = np.std(acc)
    print("\t average MEE of " + "{:.6f}%".format(mean) + " and std dev of " + "{:.2f}%".format(std))
    mean = np.mean(epochs)
    std = np.std(epochs)
    print("\t average epochs of " + "{:.6f}".format(mean) + " and std dev of " + "{:.2f}".format(std))
    suffix = "CUP_" + ts + "_" + initialization
    with open("logs/" + suffix + "_MEE.pkl", "wb") as logfile:
        pickle.dump(acc, logfile)
    with open("logs/" + suffix + "_history.pkl", "wb") as logfile:
        pickle.dump(hist, logfile)
    with open("logs/" + suffix + "_valhistory.pkl", "wb") as logfile:
        pickle.dump(val_hist, logfile)
    fig, ax = plot.subplots()
    for h in hist:
        ax.plot(h, alpha=0.25)
    #avg = [float(sum(col))/len(col) for col in zip_longest(*hist)]
    avg = np.nanmean(np.array(list(zip_longest(*hist)), dtype=float), axis=1)
    ax.plot(avg)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(suffix)
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    plot.savefig("plots/" + suffix + "_history.png")
    plot.clf()
    fig, ax = plot.subplots()
    for h in val_hist:
        ax.plot(h, alpha=0.25)
    #avg = [float(sum(col))/len(col) for col in zip_longest(*val_hist)]
    avg = np.nanmean(np.array(list(zip_longest(*val_hist)), dtype=float), axis=1)
    ax.plot(avg)
    ax.set_ylabel("Val Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(suffix)
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    plot.savefig("plots/" + suffix + "_valhistory.png")
    plot.clf()
