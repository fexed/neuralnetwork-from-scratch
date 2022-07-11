from weight_initialization import WeightInitialization, RandomUniform
from activationfunctions import ActivationFunction, Identity
import numpy as np

class FullyConnectedLayer():
    def __init__(self, in_size, out_size, activation: ActivationFunction = Identity(), init_strategy : WeightInitialization = RandomUniform()):
        self.in_size = in_size
        self.out_size = out_size
        
        self.weights, self.bias = init_strategy.generate(in_size, out_size)
        
        self.activation = activation
        self.reset_gradients()

        self.prev_weight_update = 0  # for momentum purposes
        self.prev_bias_update = 0  # for momentum purposes


    def reset_gradients(self):
        self.weights_gradient = np.zeros(self.weights.shape) #not necessary
        self.bias_gradient = np.zeros(self.bias.shape) #not necessary


    def forward_propagation(self, input, dropout=1, nesterov=0):
        self.input = input

        if nesterov != 0: 
            self.neseterov_weight_update(nesterov)

        # check how many neurons to keep
        not_dropped_units = np.random.rand(self.weights.shape[0], self.weights.shape[1]) < dropout
        not_dropped_biases = np.random.rand(self.bias.shape[0], self.bias.shape[1]) < dropout

        active_weights = np.multiply(self.weights, not_dropped_units)
        active_bias = np.multiply(self.bias, not_dropped_biases)

        self.net = np.dot(self.input, active_weights) + active_bias 
        self.output = self.activation.forward(self.net)

        return self.output


    def neseterov_weight_update(self, nesterov): 
        #Save momentum alpha_dv, gradient will be add next
        self.prev_weight_update = np.multiply(self.prev_weight_update, nesterov) 
        self.prev_bias_update = np.multiply(self.prev_bias_update, nesterov) 

        #Update the weighs before gradient computtion (Nesterov)
        self.weights += self.prev_weight_update
        self.bias += self.prev_bias_update


    #@TODO Change naming here
    def backward_propagation(self, delta):
        delta = np.multiply(self.activation.derivative(self.net), delta)
        
        input_error = np.dot(delta, self.weights.T)

        # the weights are updated according to their contribution to the error
        self.weights_gradient += np.dot(self.input.T, delta)
        self.bias_gradient += delta

        return input_error


    def update_weights(self, eta, regularizator=None, momentum = 0, nesterov=0):
        # TODO: nesterov
        dW = eta*self.weights_gradient
        dB = eta*self.bias_gradient

        # Apply the regularization/penalty term to acheive weight decay.
        if regularizator: 
            dW = dW - regularizator.derivative(self.weights)
            dB = dB - regularizator.derivative(self.bias)

        # Momentum requires saving of previous gradients, to sum them to current one.
        if momentum > 0:
            dW += momentum*self.prev_weight_update
            dB += momentum*self.prev_bias_update
            self.prev_weight_update = dW
            self.prev_bias_update = dB

        # Nesterov momentum requires saving of previous gradients, to sum them to current one.
        elif nesterov > 0:
            dW += nesterov*self.prev_weight_update
            dB += nesterov*self.prev_bias_update
            self.prev_weight_update = dW
            self.prev_bias_update = dB

        #@TODO the same thing happens both for Classical and Nesterov Momentum, maybe the code can be simplified.
        self.weights += dW
        self.bias += dB