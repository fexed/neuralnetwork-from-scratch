import numpy as np
from activationfunctions import tanh, tanh_prime, sigmoid, sigmoid_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
import matplotlib.pyplot as plot
import time
import pickle
from kfold import KFold


def compare(a, b, tollerance=1e-03):
    return abs(a - b) <= tollerance * max(abs(a), abs(b))


def test_CUP(output=True):
    ts = str(time.time()).split(".")[0]  # current timestamp for log purposes
    if (output): print("\n\n****CUP")
    cupfile = open("datasets/CUP/ML-CUP21-TR.csv", "r")
    xtr = []
    ytr = []
    for line in cupfile.readlines():
        if (line.startswith("#")):
            continue
        vals = line.split(",")
        # TODO test preprocessing
        # each feature = (feature - mean)/stddev
        # the mean array is np.mean(X, axis=0), mean for each feature column
        # the stddev array is np.std(X, axis=0), stddev for each feature
        xtr.append([[float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7]), float(vals[8]), float(vals[9]), float(vals[10])]])
        ytr.append([[float(vals[11]), float(vals[12])]])
    X = np.array(xtr)
    Y = np.array(ytr)
    if (output): print("Training set of " + str(X.size) + " elements")
    folds = 5
    # train
    net = Network("CUP " + str(folds) + "-fold test", MSE, MSE_prime)
    net.add(FullyConnectedLayer(10, 15, tanh, tanh_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(15, 10, tanh, tanh_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(10, 2, initialization_func="normalized_xavier"))
    if (output): net.summary()
    mean_accuracy = 0 #mean accuracy over the kfolds
    kfold = KFold(folds, X, Y)
    suffix = "CUP_" + ts
    fig, ax = plot.subplots()
    while (kfold.hasNext()):
        net = Network("CUP " + str(folds) + "-fold test", MSE, MSE_prime)
        net.add(FullyConnectedLayer(10, 19, tanh, tanh_prime, initialization_func="normalized_xavier"))
        net.add(FullyConnectedLayer(19, 19, tanh, tanh_prime, initialization_func="normalized_xavier"))
        net.add(FullyConnectedLayer(19, 19, tanh, tanh_prime, initialization_func="normalized_xavier"))
        net.add(FullyConnectedLayer(19, 2, initialization_func="normalized_xavier"))
        xtr, xvl, ytr, yvl = kfold.next_fold()
        history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=600, learning_rate=0.001, verbose=output, early_stopping=50)

        # accuracy on validation set
        out = net.predict(xvl)
        accuracy = 0
        print(yvl[0], " ", out[0])
        for i in range(len(out)):
            # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
            if (compare(yvl[i][0][0], out[i][0][0]) and compare(yvl[i][0][1], out[i][0][1])): accuracy += 1
        accuracy /= len(out)
        accuracy *= 100
        mean_accuracy += accuracy
        if (output): print("\n\nAccuracy on CUP validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

        ax.plot(history)
        #ax.plot(val_history)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.set_title(suffix)
        #with open("logs/" + suffix + "_history.pkl", "wb") as logfile:
            #pickle.dump(history, logfile)
        #with open("logs/" + suffix + "_valhistory.pkl", "wb") as logfile:
            #pickle.dump(val_history, logfile)

    mean_accuracy /= folds
    if (output): print("\n\nMean accuracy over " + str(folds) + " folds: {:.4f}%".format(mean_accuracy))

    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    plot.savefig("plots/" + suffix + "_" + str(folds) + "folds_history.png")
    plot.clf()
    return mean_accuracy


print("Beginning tests\n")
acc = []
#for j in range(0, 10):
#    acc.append(test_CUP(output=False))
acc.append(test_CUP(output=True))
print("CUP", end=" ")
mean = np.mean(acc)
std = np.std(acc)
print("with an accuracy of " + "{:.2f}%".format(mean) + " and std dev of " + "{:.2f}%".format(std))
