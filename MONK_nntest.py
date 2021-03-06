from activationfunctions import Sigmoid
from losses import BinaryCrossentropy
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import plot_and_save, tr_vl_split, roc_curve
from kfold import KFold
from preprocessing import one_hot_encoding
from regularizators import L2
import numpy as np
import matplotlib.pyplot as plot
from metrics import Accuracy, ROCCurve, ConfusionMatrix
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
    xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio=0.2)
    if (monk == 1):
        net = Network("MONK" + str(monk), BinaryCrossentropy(), momentum=0.8)
        net.add(FullyConnectedLayer(input_size, 20, Sigmoid(), initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 20, Sigmoid(), initialization_func="xavier"))
        net.add(FullyConnectedLayer(20, 1, Sigmoid(), initialization_func="xavier"))
        suffix += "_1L_20U_0.8M_0.1LR_xavier"
        if (output): net.summary()
        history, val_history, accuracy_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.1, verbose=output, early_stopping=50, metric = ConfusionMatrix())
    elif (monk == 2):
        net = Network("MONK" + str(monk), BinaryCrossentropy())
        net.add(FullyConnectedLayer(input_size, 10, Sigmoid(), initialization_func="normalized_xavier"))
        net.add(FullyConnectedLayer(10, 10, Sigmoid(), initialization_func="normalized_xavier"))
        net.add(FullyConnectedLayer(10, 1, Sigmoid(), initialization_func="normalized_xavier"))
        suffix += "_1L_10U_0.05LR_normxavier"
        if (output): net.summary()
        history, val_history, accuracy_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.05, verbose=output, early_stopping=50, metric = ConfusionMatrix())
    elif (monk == 3):
        net = Network("MONK" + str(monk), BinaryCrossentropy())
        net.add(FullyConnectedLayer(input_size, 10, Sigmoid(), initialization_func="xavier"))
        net.add(FullyConnectedLayer(10, 10, Sigmoid(), initialization_func="xavier"))
        net.add(FullyConnectedLayer(10, 1, Sigmoid(), initialization_func="xavier"))
        suffix += "_1L_10U_0.01LR_xavier"
        if (output): net.summary()
        history, val_history, accuracy_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.01, verbose=output, early_stopping=50, metric = ConfusionMatrix())

    # accuracy on validation set
    accuracy = Accuracy().compute(net, xvl, yvl)
    if (output): print("Accuracy on MONK" + str(monk) + " validation set of {:.4f}%".format(accuracy))

    plot_and_save(title=suffix, history=history, validation_history=val_history, ylabel="Loss", xlabel="Epochs", savefile=suffix + "_history")
    plot_and_save(title=suffix, history=[row[0] for row in accuracy_history], ylabel="Accuracy", xlabel="Epochs", savefile=suffix + "_accuracy")
    plot_and_save(title=suffix, history=[row[1] for row in accuracy_history], ylabel="Specificity", xlabel="Epochs", savefile=suffix + "_specificity")
    plot_and_save(title=suffix, history=[row[2] for row in accuracy_history], ylabel="Sensitivity", xlabel="Epochs", savefile=suffix + "_sensitivity")
    plot_and_save(title=suffix, history=[row[3] for row in accuracy_history], ylabel="Precision", xlabel="Epochs", savefile=suffix + "_precision")

    FPR, TPR, AUC = ROCCurve().compute(net, xvl, yvl)
    roc_curve(title=suffix, FPR=FPR, TPR=TPR, AUC=AUC, savefile=suffix + "_ROC2")
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
