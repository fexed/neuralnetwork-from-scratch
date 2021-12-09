import numpy as np
from activationfunctions import tanh, tanh_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network
import matplotlib.pyplot as plot


def f(x, y):
    return x+y;


def test_function():
    # TEST xor
    xtr = []
    ytr = []
    for i in range(10):
        x = i
        y = np.random.randint(10)
        xtr.append([[x, y]])
        ytr.append([[f(x, y)]])
    xtr = np.array(xtr)
    ytr = np.array(ytr)
    print("Training set")
    for i in range(10):
        print(str(xtr[i][0][0]) + " + " + str(xtr[i][0][1]) + " = " + str(ytr[i][0][0]))
    net = Network("linear test")
    net.add(FullyConnectedLayer(2, 8, tanh, tanh_prime))
    net.add(FullyConnectedLayer(8, 1))
    # train
    net.set_loss(MSE, MSE_prime)
    net.summary()
    history = net.training_loop(xtr, ytr, epochs=1000, learning_rate=0.1)

    # test
    xts = []
    yts = []
    for i in range(10, 20):
        x = i
        y = np.random.randint(20)
        xts.append([[x, y]])
        yts.append([[f(x, y)]])
    xts = np.array(xts)
    yts = np.array(yts)
    out = net.predict(xts)
    print("\n\nPredictions")
    for i in range(10):
        print(str(xts[i][0][0]) + " + " + str(xts[i][0][1]) + " = " + str(f(xts[i][0][0], xts[i][0][1])) + ", and predicted is " + str(out[i][0][0]))

    plot.plot(history)
    suffix = "linear_x+y"
    plot.savefig("/home/fexed/ML/fromscratch/plots/" + suffix + "_history.png")

test_function()
