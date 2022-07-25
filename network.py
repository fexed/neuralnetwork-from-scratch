from hyperparameter import Momentum, Dropout, Momentum
from regularizators import L2


class Network(): 
    def __init__(self, loss, regularization=L2(0), momentum=Momentum(0), dropout=Dropout(1)):
        self.loss = loss
        self.regularization = regularization

        self.momentum = momentum
        self.dropout = dropout

        self.layers = []


    def add(self, layer): 
            self.layers.append(layer)


    def forward_propagation(self, pattern, inference=False):
        """ Performs the forward propagation of the network """
        output = pattern
        if inference: 
            for layer in self.layers: 
                output = layer.forward_propagation(output)
        else: #It's training
            for layer in self.layers:
                output = layer.forward_propagation(output, self.dropout.rate, self.momentum.alpha, self.momentum.nesterov) 
        
        return output


    def backward_propagation(self, output, target):
        """ Performs the backward propagation of the network """

        gradient = self.loss.derivative(output, target)

        for layer in reversed(self.layers):
            gradient = layer.backward_propagation(gradient)
        
        return gradient

     
    def update_weights(self, learning_rate):
        for layer in self.layers:
                layer.update_weights(learning_rate, self.regularization, self.momentum.alpha)


    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()
    