import numpy as np
from math import sqrt


class Layer:
    # simple empty layer structure
    def __init__(self):
        self.input = None
        self.output = None


    # computes the output of the layer for a given input
    def forward_propagation(self, input):
        raise NotImplementedError


    # computes the delta-error over the input for a given delta-error over the
    # output and updates any parameter
    def backward_propagation(self, output_error, learning_rate, momentum = 0, regularizator=None):
        raise NotImplementedError


class FullyConnectedLayer(Layer):
    # a simple layer, linear or with an activation function
    # in_size = number of input neurons
    # out_size = number of output neurons
    def __init__(self, in_size, out_size, activation = None, activation_prime = None, initialization_func = None):
        if not(initialization_func is None):
            if (initialization_func == "xavier"):  # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=each%20in%20turn.-,Xavier%20Weight%20Initialization,-The%20xavier%20initialization
                l, u = -(1.0 / sqrt(in_size)), (1.0 / sqrt(in_size))
                self.weights = np.random.uniform(low=l, high=u, size=(in_size, out_size))
                self.bias = np.random.uniform(low=l, high=u, size=(1, out_size))
            elif (initialization_func == "normalized_xavier"):  # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=to%20One%20Hundred-,Normalized%20Xavier%20Weight%20Initialization,-The%20normalized%20xavier
                l, u = -(6.0 / sqrt(in_size + out_size)), (6.0 / sqrt(in_size + out_size))
                self.weights = np.random.uniform(low=l, high=u, size=(in_size, out_size))
                self.bias = np.random.uniform(low=l, high=u, size=(1, out_size))
            elif (initialization_func == "he"):  # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=on%20ImageNet%20Classification.%E2%80%9D-,He%20Weight%20Initialization,-The%20he%20initialization
                # for ReLU
                dev = sqrt(2.0 / in_size)
                self.weights = np.random.normal(loc=0.0, scale=dev, size=(in_size, out_size))
                self.bias = np.random.normal(loc=0.0, scale=dev, size=(1, out_size))
            elif (initialization_func == "basic"):
                self.bias = [np.full(out_size, 0)]
                self.weights = np.random.uniform(-1/in_size, 1/in_size, (in_size, out_size))
        else:
            self.weights = np.random.rand(in_size, out_size) - 0.5  # so to have few <0 and few >0
            self.bias = np.random.rand(1, out_size) - 0.5  # so to have few <0 and few >0
        self.activation = activation
        self.activation_prime = activation_prime
        self.prev_weight_update = 0  # for momentum purposes
        self.prev_bias_update = 0  # for momentum purposes

    def get_weights(self):
        return self.weights

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias  # net output
        if not(self.activation is None):
            # the activation function is optional
            self.activation_input = self.output
            self.output = self.activation(self.output)
        return self.output


    def backward_propagation(self, gradient, eta, momentum = 0, regularizator_l=0):
        if not(self.activation_prime is None):
            # if there's activation function specified, then we compute its
            # derivative
            gradient = np.multiply(self.activation_prime(self.activation_input), gradient)
        # the weights are updated according to their contribution to the error
        weights_update = eta * np.dot(self.input.T, gradient)
        bias_update = eta * gradient

        # regularization
        weights_update -= np.multiply(self.weights, regularizator_l)
        bias_update -= np.multiply(self.bias, regularizator_l)

        if (momentum > 0):
            # with momentum we consider the previous update too
            weights_update += np.multiply(self.prev_weight_update, momentum)
            bias_update += np.multiply(self.prev_bias_update, momentum)

            # store this update for the next backprop in this layer
            self.prev_weight_update = weights_update
            self.prev_bias_update = bias_update

        # the basic parameter update
        self.weights -= weights_update
        self.bias -= bias_update

        input_error = np.dot(gradient, self.weights.T)
        return input_error


class ActivationLayer(Layer):
    # a layer with just an activation function
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime


    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation(self.input)  # simple value of the activation
        return self.output


    def backward_propagation(self, gradient, eta, momentum = 0, regularizator=None):
        # simple derivative of the activation function
        return np.multiply(self.activation_prime(self.input), gradient)
