import numpy as np
from activationfunctions import tanh, tanh_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network
import matplotlib.pyplot as plot


def test_xor():
    # TEST xor
    x_train = np.array([ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ])
    y_train = np.array([ [ [0] ], [ [1] ], [ [1] ], [ [0] ] ])
    net = Network("XOR test")
    net.add(FullyConnectedLayer(2, 3, tanh, tanh_prime))
    net.add(FullyConnectedLayer(3, 1, tanh, tanh_prime))
    # train
    net.set_loss(MSE, MSE_prime)
    net.summary()
    history = net.training_loop(x_train, y_train, epochs=1000, learning_rate=0.1)

    # test
    out = net.predict(x_train)
    print("Pred\tTrue")
    for i in range(4):
        print("{:.4f}".format(out[i][0][0]) + "\t" + str(y_train[i][0][0]), end = "")
        if (out[i][0][0] < 0.5 and y_train[i][0][0] == 0):
            print("\tOK")
        elif (out[i][0][0] > 0.5 and y_train[i][0][0] == 1):
            print("\tOK")
        else:
            print("\tNO")

    plot.plot(history)
    suffix = "XOR"
    plot.savefig("plots/" + suffix + "_history.png")

test_xor()
