from activationfunctions import sigmoid, sigmoid_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from kfold import KFold
from preprocessing import continuous_standardizer
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plot
import time
import pickle


def compare(a, b, tollerance=1e-03):
    return abs(a - b) <= tollerance * max(abs(a), abs(b))


def test_CUP(output=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    if (output): print("\n\n****CUP")
    X, Y = load_cup(verbose=True, test=False)
    # train
    X, n_min, n_max = min_max_normalizer(X)
    X, means, std = continuous_standardizer(X)
    xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)
    suffix = "CUP_" + ts
    net = Network("CUP test", MSE, MSE_prime, regularizator=L2, regularization_l=0.15, momentum=0)
    net.add(FullyConnectedLayer(10, 29, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(29, 29, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(29, 29, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(29, 29, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(29, 29, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(29, 2, initialization_func="normalized_xavier"))
    if (output): net.summary()
    history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.001, verbose=output, early_stopping=150, batch_size=1)

    # accuracy on validation set
    out = net.predict(xvl)
    accuracy = 0
    for i in range(len(out)):
        # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
        if (compare(yvl[i][0][0], out[i][0][0]) and compare(yvl[i][0][1], out[i][0][1])): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100

    if (output): print("Accuracy: {:.4f}%".format(accuracy))

    fig, ax = plot.subplots()
    ax.plot(history)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(suffix)
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    plot.savefig("plots/" + suffix + "_history.png")
    plot.clf()


print("Beginning tests\n")
acc = []
#for j in range(0, 10):
#    acc.append(test_CUP(output=False))
acc.append(test_CUP(output=True))
mean = np.mean(acc)
std = np.std(acc)
print("CUP with an accuracy of " + "{:.2f}%".format(mean) + " and std dev of " + "{:.2f}%".format(std))
