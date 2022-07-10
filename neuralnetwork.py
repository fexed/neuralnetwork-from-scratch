from turtle import forward
import numpy as np
import pickle
import random
from utils import training_progress

class Network:
    """ Base class for the neural networks used in this project """

    def __init__(self, name="-unnamed-", loss=None, regularizator=None, momentum=0, dropout=0, nesterov=0):
        """ Initializes the neural network with some parameters

        Parameters
        ----------

        name : str, optional
            The name of the neural network, used for logging and output purposes
        loss : Loss, optional
            The loss function to be used during training
        regularizator : Regularizator, optional
            The regularizator to be applied during training
        momentum = float, optional
            The momentum of the training
        nesterov = boolean, option
            #@TODO change this doc line
            The way to apply momentum heuristic: Nesterov or "Classical"
        dropout = float, optional
            The dropout percentage
        """

        self.name = name  # for logging and output purposes
        self.layers = []  # all the layers will be stored here
        self.loss = loss
        self.regularizator = regularizator
        self.momentum = momentum
        self.dropout = 1 - dropout  # prob of keeping the neuron
        self.nesterov = nesterov


    def summary(self):
        """ A summary of the network, for logging and output purposes """

        trainable_parameters = 0  # output purposes
        print("Neural Network \"" + self.name + "\"")
        print("+==== Structure")
        nlayers = len(self.layers)
        try:
            print("|IN\t" + type(self.layers[0]).__name__ + ": " + str(len(self.layers[0].weights)) + " units" , end = "")
            trainable_parameters += self.layers[0].weights.size
            trainable_parameters += self.layers[0].bias.size
            print(" with " + self.layers[0].activation.name + " activation function", end = "")
        except AttributeError:
            pass
        print("")
        for i in range(nlayers-1):
            try:
                print("|HID\t" + type(self.layers[i]).__name__ + ": " + str(self.layers[i].weights[0].size) + " units" , end = "")
                trainable_parameters += self.layers[i].weights.size
                trainable_parameters += self.layers[i].bias.size
                print(" with " + self.layers[i].activation.name + " activation function", end = "")
            except AttributeError:
                pass
            print("")
        try:
            print("|OUT\t" + type(self.layers[-1]).__name__ + ": " + str(self.layers[-1].weights[0].size) + " units" , end = "")
            trainable_parameters += self.layers[-1].weights.size
            trainable_parameters += self.layers[-1].bias.size
            print(" with " + self.layers[-1].activation.name + " activation function", end = "")
        except AttributeError:
            pass
        print("")
        if not(self.loss is None):
            print("+==== Loss: " + self.loss.name, end="")
        else:
            print("+====")
        if not(self.regularizator is None):
            print(" and " + self.regularizator.name + " regularizator with lambda = " + str(self.regularizator.l), end="")
        if (self.momentum > 0):
            print(" and momentum = " + str(self.momentum), end="")
        if (self.dropout < 1):
            print(" and dropout = " + str(self.dropout), end="")
        if (self.nesterov):
            print(" and Nesterov momentum", end="")
        print("") 
        print("For a total of " + str(trainable_parameters) + " trainable parameters")


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
        
        N = len(X_TR)
        tr_loss_hist = []
        val_loss_hist = []
        tr_metric_hist = []
        val_metric_hist = []

        #Move them outta here.
        if (verbose):
            print("Beginning training loop with " + str(epochs) + " targeted epochs over " + str(N) + " training elements and learning rate = " + str(learning_rate), end="")
            if (batch_size > 1):
                print(" (batch size = " + str(batch_size) + ")", end="")
            if not(early_stopping is None):
                print(", with early stopping = " + str(early_stopping), end="")
            if len(X_VAL) != 0:
                print(" and validation set present", end="")
            if not(lr_decay is None):
                print(", with " + str(lr_decay))
            if not(metric is None):
                print(". The evaluation metric is " + metric.name, end="")
            print("")   

        es_epochs = 0  # counting early stopping epochs if needed
        min_error = float('inf') #@TODO Check why this is done

        #TRAINING ACTUALLY BEGINS HERE.

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
                
            #@TODO Reformat using if into the f string
            if (verbose): 
                training_progress(i+1, epochs, suffix=f"loss = {tr_loss_hist[i]}%, val_loss = {val_loss_hist[i]}%")

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
                        if (verbose): 
                             #@TODO Reformat using if into the f string
                            print('\nEarly stopping on epoch %d of %d with loss = %f and val_loss = %f' % (i+1, epochs, tr_loss_hist[i], val_loss_hist[i]))
                        break
                else:
                    # we're good
                    es_epochs = 0
                    min_error = check_error
        if (verbose): print("")
        return tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist 
    

    def savenet(self, filename):
        """ Saves the neural network in a pickle """

        with open(filename, "wb") as savefile:
            pickle.dump(self.__dict__, savefile)


    def loadnet(self, filename):
        """ Loads the neural network from a pickle """

        with open(filename, "rb") as savefile:
            newnet = pickle.load(savefile)

        self.__dict__.clear()  # clear current net
        self.__dict__.update(newnet)
