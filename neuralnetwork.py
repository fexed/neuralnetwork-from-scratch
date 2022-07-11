import numpy as np
from logger import MLPLogger

class Network():
    """ Base class for the neural networks used in this project """

    def __init__(self, name = '', loss=None, regularizator=None, momentum=0, dropout=0, nesterov=0):
        self.name = name
        self.layers = []  # all the layers will be stored here
        self.loss = loss
        self.regularizator = regularizator
        self.momentum = momentum
        self.dropout = 1 - dropout  # prob of keeping the neuron
        self.nesterov = nesterov
        
        self.logger = MLPLogger(self, [], True)

    def __call__(self, **args):
        self.training_loop(args)

    def add(self, layer):
        """ Adds another layer at the bottom of the network """
        self.layers.append(layer)


    def predict(self, data):
        """ Applies the network to the data and returns the computed values """

        N = len(data)
        results = []

        for i in range(N):
            output = data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)

        return results

    def training_epoch(self, X, Y, batch_size, eta): 
        batches_X = [X[i:i+batch_size] for i in range(0, len(X), batch_size)] 
        batches_Y = [Y[i:i+batch_size] for i in range(0, len(Y), batch_size)] 

        self.reset_gradient() #reset the gradient

        batch_gradient = 0
        for batch_X, batch_Y in zip(batches_X, batches_Y):
            for pattern, target in zip(batch_X, batch_Y):
                pattern_output = self.forward_propagation(pattern)
                batch_gradient += self.backward_propagation(pattern_output, target)

            self.update_weights(eta) # apply backprop and delta rule to update weights 


    def forward_propagation(self, p):
        """ Performs the forward propagation of the network """
        output = p
        for layer in self.layers:
            output = layer.forward_propagation(output, self.dropout)

        return output


    def backward_propagation(self, output, target):
        """ Performs the backward propagation of the network """

        gradient = self.loss.derivative(output, target)

        for layer in reversed(self.layers):
            gradient = layer.backward_propagation(gradient)
        
        return gradient


    def reset_gradient(self):
        for layer in self.layers:
            layer.reset_gradient()


    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate, regularizator=self.regularizator, momentum=self.momentum, nesterov=self.nesterov)


    def training_loop(self, X_TR, Y_TR, X_VAL=[], Y_VAL=[], epochs=1000, learning_rate=0.01, early_stopping=None, batch_size=1, lr_decay=None, metric=None, verbose=True):
        
        self.logger = MLPLogger(verbose, [], False)
        
        N = len(X_TR)
        tr_loss_hist = []
        val_loss_hist = []
        tr_metric_hist = []
        val_metric_hist = []

        es_epochs = 0  # counting early stopping epochs if needed
        min_error = float('inf') #@TODO Check why this is done

        for i in range(epochs):
            # Training happens here
            self.training_epoch(X_TR, Y_TR, batch_size, learning_rate)

            # Should be checked!
            if not (lr_decay is None):
                if (lr_decay.type == "linear"):
                    if learning_rate > lr_decay.final_value:
                        lr_decay_alpha = i/lr_decay.last_step
                        learning_rate = (1 - lr_decay_alpha) * learning_rate + lr_decay_alpha * lr_decay.final_value

            # Compute learning curves 
            tr_output = self.predict(X_TR)  #@TODO Should we calculate after weight update or reuse outputs from the forward propagation part?
            val_output = self.predict(X_VAL)

            tr_loss_hist.append(self.loss.compute(tr_output, Y_TR))
            tr_metric_hist.append(metric.compute(tr_output, Y_TR))
            
            val_loss_hist.append(self.loss.compute(val_output, Y_VAL))
            val_metric_hist.append(metric.compute(val_output, Y_VAL))
    
            self.logger.training_progress(i, epochs, tr_loss_hist[i], val_loss_hist[i])

            #@TODO Check Early Stopping Implementation
            if not(early_stopping is None):
                # with early stopping we need to check the current situation and
                # stop if needed
                check_error = val_loss_hist[i] or tr_loss_hist[i]

                if check_error >= min_error or np.isnan(check_error):
                    # the error is increasing or is stable, or there was an
                    # overflow situation, hence we are going toward an ES
                    es_epochs += 1
                    if es_epochs == early_stopping:
                        self.logger.early_stopping_log()
                        break
                else:
                    # we're good
                    es_epochs = 0
                    min_error = check_error

        return tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist 
    