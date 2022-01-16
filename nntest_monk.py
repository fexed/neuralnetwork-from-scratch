from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import plot_loss
from kfold import KFold
from preprocessing import one_hot_encoding
from regularizators import L2
import numpy as np
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import time
import pickle


def test_MONK(monk=1, output=True, use_one_hot_encoding=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
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

    input_size = 6

    if use_one_hot_encoding:
        X, input_size = one_hot_encoding(X)

    # train
    folds = 1
    suffix = "MONK" + str(monk) + "_" + ts
    fig, ax = plot.subplots()
    xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)
    if (monk == 1):
        net = Network("MONK" + str(monk), binary_crossentropy, binary_crossentropy_prime, momentum=0.8)
        net.add(FullyConnectedLayer(input_size, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 1, sigmoid, sigmoid_prime, initialization_func="xavier"))
        suffix += "_1L_20U_0.8M_xavier"
        net.summary()
        history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.1, verbose=output, early_stopping=50)
    elif (monk == 2):
        net = Network("MONK" + str(monk), binary_crossentropy, binary_crossentropy_prime, momentum=0.8)
        net.add(FullyConnectedLayer(input_size, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 1, sigmoid, sigmoid_prime, initialization_func="xavier"))
        suffix += "_1L_20U_0.8M_xavier"
        net.summary()
        history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.1, verbose=output, early_stopping=50)
    elif (monk == 3):
        net = Network("MONK" + str(monk), binary_crossentropy, binary_crossentropy_prime, momentum=0.8)
        net.add(FullyConnectedLayer(input_size, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 20, sigmoid, sigmoid_prime, initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 1, sigmoid, sigmoid_prime, initialization_func="xavier"))
        suffix += "_1L_20U_0.8M_xavier"
        net.summary()
        history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.1, verbose=output, early_stopping=50)

    # accuracy on validation set
    out = net.predict(xvl)
    accuracy = 0
    for i in range(len(out)):
        val = 0 if out[i].item() < 0.5 else 1  # "normalizing" output
        if (yvl[i].item() == val): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100
    if (output): print("Accuracy on MONK" + str(monk) + " validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

    plot_loss(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
    return accuracy


print("Beginning tests")
for i in range(1, 4):
    acc = []
    for j in range(0, 1):
        acc.append(test_MONK(i, output=True))
        #print(str(j+1), end=" ", flush=True)
    print("")
    print("MONK" + str(i), end=" ")
    mean = np.mean(acc)
    std = np.std(acc)
    print("with an accuracy of " + "{:.2f}%".format(mean) + " and std dev of " + "{:.2f}%".format(std))
