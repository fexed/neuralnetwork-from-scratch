import numpy as np
from math import sqrt


class Layer:
    """ Simple empty layer structure """

    def __init__(self):
        """ Initializes the layer with its parameters """

        self.input = None
        self.output = None


    def forward_propagation(self, input, dropout=1):
        """ Computes the output of the layer for a given input

        Parameters
        ----------
        input
            The input of the layer
        dropout : float
            Percentage of neurons to keep
        """

        raise NotImplementedError


    def backward_propagation(self, output_error, learning_rate, momentum = 0, regularizator=None, nesterov=False):
        """ Computes the delta-error over the input for a given delta-error over
        the output and updates any parameter

        Parameters
        ----------
        output_error
            The gradient to use for the SGD
        learning_rate : float
            The learning rate to use
        momentum : float, optional
            The rate of momentum
        regularizator : Regularizator
            The regularizator to use
        nesterov: bool,
            The way to apply momentum heuristic: Nesterov or "Classical"
        """

        raise NotImplementedError


class FullyConnectedLayer(Layer):
    """ Simple layer, linear or with an activation function"""

    def __init__(self, in_size, out_size, activation = None, initialization_func = None):
        """ Initializes the layer with its dimensions. Can also specify the
        activation function and the initialization function of its neurons.

        Parameters
        ----------
        in_size : int
            Number of input neurons
        out_size : int
            Number of output neurons
        activation : ActivationFunction, optional
            The activation function of this layer
        initialization_func : str, optional
            The type of initialization of the neurons of this layer.
            Supported: [None, "xavier", "normalized_xavier", "he", "basic"]
        """
        self.activation = activation
        self.prev_weight_update = 0  # for momentum purposes
        self.prev_bias_update = 0  # for momentum purposes
        self.in_size = in_size
        self.out_size = out_size
        self.init_func = initialization_func


    def init_weights(self):
        if not(self.init_func is None):
            if (self.init_func == "xavier"):  # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=each%20in%20turn.-,Xavier%20Weight%20Initialization,-The%20xavier%20initialization
                l, u = -(1.0 / sqrt(self.in_size)), (1.0 / sqrt(self.in_size))
                self.weights = np.random.uniform(low=l, high=u, size=(self.in_size, self.out_size))
                self.bias = np.random.uniform(low=l, high=u, size=(1, self.out_size))
            elif (self.init_func == "normalized_xavier"):  # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=to%20One%20Hundred-,Normalized%20Xavier%20Weight%20Initialization,-The%20normalized%20xavier
                l, u = -(6.0 / sqrt(self.in_size + self.out_size)), (6.0 / sqrt(self.in_size + self.out_size))
                self.weights = np.random.uniform(low=l, high=u, size=(self.in_size, self.out_size))
                self.bias = np.random.uniform(low=l, high=u, size=(1, self.out_size))
            elif (self.init_func == "he"):  # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/#:~:text=on%20ImageNet%20Classification.%E2%80%9D-,He%20Weight%20Initialization,-The%20he%20initialization
                # for ReLU
                dev = sqrt(2.0 / self.in_size)
                self.weights = np.random.normal(loc=0.0, scale=dev, size=(self.in_size, self.out_size))
                self.bias = np.random.normal(loc=0.0, scale=dev, size=(1, self.out_size))
            elif (self.init_func == "basic"):
                self.bias = [np.full(self.out_size, 0)]
                self.weights = np.random.uniform(-1/self.in_size, 1/self.in_size, (self.in_size, self.out_size))
        else:
            self.weights = np.random.rand(self.in_size, self.out_size) - 0.5  # so to have few <0 and few >0
            self.bias = np.random.rand(1, self.out_size) - 0.5  # so to have few <0 and few >0


    def get_weights(self):
        """ Simple getter for the weights of this layer

        Returns
        -------
        weights
            The weights of this layer
        """

        return self.weights


    def forward_propagation(self, input, dropout=1, nesterov=0):
        """ Computes the output of the layer for a given input

        Parameters
        ----------
        input
            The input of the layer
        dropout : float
            Percentage of neurons to keep
        """

        self.input = input

        if nesterov != 0: 
            #Save momentum alpha_dv, gradient will be add next
            self.prev_weight_update = np.multiply(self.prev_weight_update, nesterov) 
            self.prev_bias_update = np.multiply(self.prev_bias_update, nesterov) 

            #Update the weighs before gradient computtion (Nesterov)
            self.weights += self.prev_weight_update
            self.bias += self.prev_bias_update
            
        # TODO: check overflow situations

        # check how many neurons to keep
        keep = np.random.rand(self.weights.shape[0], self.weights.shape[1]) < dropout
        newweights = np.multiply(self.weights, keep)
        keep = np.random.rand(self.bias.shape[0], self.bias.shape[1]) < dropout
        newbias = np.multiply(self.bias, keep)
        self.output = np.dot(self.input, newweights) + newbias  # net output
        if not(self.activation is None):
            # the activation function is optional
            # without it the output value is linear
            self.activation_input = self.output
            self.output = self.activation.forward(self.output)
        return self.output


    def zero_gradient(self):
        self.bias_gradient = 0
        self.weights_gradient = np.zeros(self.weights.shape)


    def backward_propagation(self, gradient):
        """ Computes the delta-error over the input for a given delta-error over
        the output

        Parameters
        ----------
        gradient
            The gradient to use for the SGD
        """

        if not(self.activation is None):
            # if there's activation function specified, then we compute its
            # derivative
            gradient = np.multiply(self.activation.derivative(self.activation_input), gradient)
        
        input_error = np.dot(gradient, self.weights.T)

        # the weights are updated according to their contribution to the error
        self.weights_gradient += np.dot(self.input.T, gradient)
        self.bias_gradient += gradient

        return input_error


    def update_weights(self, eta, regularizator=None, momentum = 0, nesterov=False):
        """ Updates the weights of the layer according to the gradient computed
        during the backward propagation

        Parameters
        ----------
        eta : float
            The learning rate to use
        momentum : float, optional
            The rate of momentum
        nesterov: boolean
            Flag used to save pervious weight update when using Nesterov momentum
            In such a case the 'momentum' parameter MUST be 0 (or unset).
        """

        # TODO: nesterov

        dW = eta*self.weights_gradient
        dB = eta*self.bias_gradient

        if  regularizator: 
            dW = dW + regularizator.derivative(self.weights)
            dB = dB + regularizator.derivative(self.bias)

        if momentum > 0:
            dW += momentum*self.prev_weight_update
            dB += momentum*self.prev_bias_update
            self.prev_weight_update = dW
            self.prev_bias_update = dB

        self.weights += dW
        self.bias += dB