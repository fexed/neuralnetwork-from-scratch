import numpy as np
from activationfunctions import softmax, softmax_deriv, tanh, tanh_prime
from losses import binary_crossentropy, binary_crossentropy_deriv, MSE, MSE_deriv
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network
from regularizators import L2
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split


def test_MONK(monk=1, output=True):
    if (output): print("\n\n****MONK" + str(monk))
    monkfile = open("/home/fexed/ML/fromscratch/datasets/MONK/monks-" + str(monk) + ".train", "r")
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
    net = Network("MONK" + str(monk) + " test", MSE, MSE_deriv)
    net.add(FullyConnectedLayer(6, 10, tanh, tanh_prime))
    net.add(FullyConnectedLayer(10, 1, tanh, tanh_prime))
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
    return accuracy

    # test set
    #monkfile = open("/home/fexed/ML/fromscratch/datasets/MONK/monks-" + str(monk) + ".test", "r")
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


    plot.plot(history)
    plot.plot(val_history)
    suffix = "MONK" + str(monk) + "_{:.2f}%".format(accuracy)
    plot.savefig("/home/fexed/ML/fromscratch/plots/" + suffix + "_history.png")
    plot.clf()


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
