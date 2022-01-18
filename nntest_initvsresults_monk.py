import numpy as np
from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network
from regularizators import L2
from utils import tr_vl_split
import matplotlib.pyplot as plot
import time
import pickle
from itertools import zip_longest


def test_MONK(monk=1, output=True, initialization="normalized_xavier"):
    if (output): print("\n\n****MONK" + str(monk))
    monkfile = open("datasets/MONK/monks-" + str(monk) + ".train", "r")
    xtr = []
    ytr = []
    for line in monkfile.readlines():
        vals = line.split(" ")
        xtr.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
        ytr.append([[int(vals[1])]])
    X = np.array(xtr)
    Y = np.array(ytr)
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio=0.2)
    if (output): print("Training set of " + str(X.size) + " elements")
    net = Network("MONK" + str(monk) + " test", binary_crossentropy, binary_crossentropy_prime, regularizator=L2, regularization_l=0.005, momentum=0.5)
    net.add(FullyConnectedLayer(6, 10, sigmoid, sigmoid_prime, initialization_func=initialization))
    net.add(FullyConnectedLayer(10, 1, sigmoid, sigmoid_prime, initialization_func=initialization))
    # train
    if (output):
        net.summary()
    history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.01, verbose=output, early_stopping=25)

    # accuracy on validation set
    out = net.predict(xvl)
    accuracy = 0
    for i in range(len(out)):
        val = 0 if out[i].item() < 0.5 else 1  # "normalizing" output
        if (yvl[i].item() == val): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100
    if (output): print("\n\nAccuracy on MONK" + str(monk) + " validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

    return accuracy, history, val_history


for i in range(1, 4):
    for initialization in ["xavier", "normalized_xavier", "he"]:
        print("Beginning tests on MONK" + str(i) + " with " + initialization)
        ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
        acc, hist, val_hist, epochs = [], [], [], []
        for j in range(0, 100):
            accuracy, history, val_history = test_MONK(i, output=False, initialization=initialization)
            acc.append(accuracy)
            hist.append(history)
            val_hist.append(val_history)
            epochs.append(len(history))
            print(str(j+1), end=" ", flush=True)
        print("")
        print("MONK" + str(i) + " with " + initialization)
        mean = np.mean(acc)
        std = np.std(acc)
        print("\t average accuracy of " + "{:.2f}%".format(mean) + " and std dev of " + "{:.2f}%".format(std))
        mean = np.mean(epochs)
        std = np.std(epochs)
        print("\t average epochs of " + "{:.2f}".format(mean) + " and std dev of " + "{:.2f}".format(std))
        suffix = "MONK" + str(i) + "_" + ts + "_" + initialization
        with open("logs/" + suffix + "_accuracy.pkl", "wb") as logfile:
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
