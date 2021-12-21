import numpy as np
from activationfunctions import relu, relu_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plot


def test_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')# / 255
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train.reshape(y_train.shape[0], 1, 10)

    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')# / 255
    y_test = np_utils.to_categorical(y_test)
    y_test = y_test.reshape(y_test.shape[0], 1, 10)

    net = Network("MNIST test")
    net.add(FullyConnectedLayer(28*28, 100, relu, relu_prime))
    #net.add(ActivationLayer(relu, relu_prime))
    net.add(FullyConnectedLayer(100, 50, relu, relu_prime))
    net.add(FullyConnectedLayer(50, 10, relu, relu_prime))
    #net.add(ActivationLayer(relu, relu_prime))

    net.set_loss(MSE, MSE_prime)
    net.summary()
    history = net.training_loop(x_train[0:1000], y_train[0:1000], epochs = 100, learning_rate = 0.01, batch_size = 100, early_stopping = 15)

    out = net.predict(x_test[0:10])
    print("Pred\tTrue")
    for i in range(10):
        print(" " + str(np.argmax(out[i])) + "\t " + str(np.argmax(y_test[i])))

    plot.plot(history)
    suffix = "MNIST"
    plot.savefig("plots/" + suffix + "_history.png")


test_mnist()
