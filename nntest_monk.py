import numpy as np
from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network
from regularizators import L2
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
import time
import pickle


def test_MONK(monk=1, output=True):
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
    xtr, xvl, ytr, yvl = train_test_split(X, Y, test_size=0.2, random_state=42)
    if (output): print("Training set of " + str(X.size) + " elements")
    net = Network("MONK" + str(monk) + " test", binary_crossentropy, binary_crossentropy_prime)
    net.add(FullyConnectedLayer(6, 10, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    net.add(FullyConnectedLayer(10, 1, sigmoid, sigmoid_prime, initialization_func="normalized_xavier"))
    # train
    if (output):
        net.summary()
    history, val_history = net.training_loop(xtr, ytr, X_validation=xvl, Y_validation=yvl, epochs=1000, learning_rate=0.01, verbose=output)

    # accuracy on validation set
    out = net.predict(xvl)
    accuracy = 0
    for i in range(len(out)):
        val = 0 if out[i].item() < 0.5 else 1  # "normalizing" output
        if (yvl[i].item() == val): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100
    if (output): print("\n\nAccuracy on MONK" + str(monk) + " validation set of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

    # test set
    #monkfile = open("datasets/MONK/monks-" + str(monk) + ".test", "r")
    #xts = []
    #yts = []
    #for line in monkfile.readlines():
        #vals = line.split(" ")
        #xts.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
        #yts.append([[int(vals[1])]])
    #xts = np.array(xts)
    #yts = np.array(yts)
    #out = net.predict(xts)
    #accuracy = 0
    #for i in range(len(out)):
        #val = 0 if out[i].item() < 0.5 else 1  # "normalizing" output
        #if (yts[i].item() == val): accuracy += 1
    #accuracy /= len(out)
    #accuracy *= 100
    #print("\n\nAccuracy on MONK" + str(monk) + " of {:.4f}%".format(accuracy) + " over " + str(len(out)) + " elements")

    suffix = "MONK" + str(monk) + "_" + ts

    fig, ax = plot.subplots()
    ax.plot(history)
    ax.plot(val_history)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(suffix)
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    plot.savefig("plots/" + suffix + "_history.png")
    plot.clf()
    with open("logs/" + suffix + "_history.pkl", "wb") as logfile:
        pickle.dump(history, logfile)
    with open("logs/" + suffix + "_valhistory.pkl", "wb") as logfile:
        pickle.dump(val_history, logfile)

    # To read, example follows:
    # with open(filename, "rb") as logfile:
    #   list = pickle.load(logfile)

    #    plot.plot(history)
    #    plot.plot(val_history)
    #    suffix = "MONK" + str(monk) + "_{:.2f}%".format(accuracy)
    #    plot.savefig("plots/" + suffix + "_history.png")
    #    plot.clf()

    return accuracy


print("Beginning tests")
for i in range(1, 4):
    acc = []
    for j in range(0, 10):
        acc.append(test_MONK(i, output=False))
        print(str(j+1), end=" ", flush=True)
    print("")
    print("MONK" + str(i), end=" ")
    mean = np.mean(acc)
    std = np.std(acc)
    print("with an accuracy of " + "{:.2f}%".format(mean) + " and std dev of " + "{:.2f}%".format(std))
