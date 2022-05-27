import numpy as np
import pickle
import random
from utils import training_progress


class Network:
    """ Base class for the neural networks used in this project """

    def __init__(self, name="-unnamed-", loss=None, regularizator=None, momentum=0, dropout=0, nesterov=False):
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


    def set_loss(self, loss):
        """ Changes the losses of the net """

        # TODO delete this?
        self.loss = loss


    def add(self, layer):
        """ Adds another layer at the bottom of the network """

        # TODO check input-output dimensions?
        self.layers.append(layer)
        self.layers[-1].init_weights()


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


    def training_loop(self, X, Y, X_validation=None, Y_validation=None, epochs=1000, learning_rate=0.01, early_stopping=None, batch_size=1, lr_decay=None, lr_decay_finalstep=500, lr_final=0.00001, metric=None, verbose=True):
        """ The main training loop for the network

        Parameters
        ----------
        X
            The features of the training set
        Y
            The labels of the training set
        X_validation : optional
            The features of the validation set
        Y_validation : optional
            The labels of the validation set
        epochs : int, optional
            The targeted epochs of the training
        learning_rate : float, optional
            The learning rate
        early_stopping: int, optional
            The number epochs of no improvement after which the training stops
        batch_size : int, optional
            The batch size of the training
        lr_decay : str, optional
            The learning rate decay strategy
        lr_decay_finalstep : int, optional
            The epoch from which the learning rate will be constant
        lr_final : float, optional
            The final learning rate
        metric : Metric, optional
            Evaluation metric to be plotted
        verbose : bool, optional
            Wether to be verbose or not
        """

        N = len(X)
        history = []  # for logging purposes
        M = 0
        if not(X_validation is None): #If validation set is provided
            val_history = []
            M = len(X_validation)
        else:
            val_history = None  # used to check if validation set is present

        if not(metric is None):
            metric_history = []

        if not(lr_decay is None):
            initial_learning_rate = learning_rate

        if (verbose):
            print("Beginning training loop with " + str(epochs) + " targeted epochs over " + str(N) + " training elements", end="")
            if (batch_size > 1):
                print("(batch size = " + str(batch_size) + ")", end="")
            if not(early_stopping is None):
                print(", with early stopping = " + str(early_stopping), end="")
            if not(val_history is None):
                print(" and validation set present", end="")
            if not(lr_decay is None):
                print(", with " + str(lr_decay) + " weight decay until epoch " + str(lr_decay_finalstep), end="")
            if not(metric is None):
                print(". The evaluation metric is " + metric.name, end="")
            print("")

        es_epochs = 0  # counting early stopping epochs if needed
        min_error = float('inf')
        for i in range(epochs):
            error = 0
            outputs = []
            targets = []
            #shuffle order of inputs each epoch
            temp = list(zip(X, Y))
            random.shuffle(temp)
            X, Y = zip(*temp)
            for j in range(N):
                # compute the output iteratively through each layer
                output = X[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output, dropout=self.dropout)
                error += self.loss.forward(Y[j], output)
                if not(self.regularizator is None):
                    regloss = 0
                    for layer in self.layers:
                        regloss += self.regularizator.forward(layer.weights)
                    error += regloss
                # compute the gradient through each layer, while applying
                # the backward propagation algorithm
                outputs.append(output)
                targets.append(Y[j])
                if ((j+1) % batch_size == 0) or (j == N-1):
                    gradient = 0
                    for k in range(len(outputs)):
                        gradient += self.loss.derivative(targets[k], outputs[k])
                    gradient /= len(outputs)
                    for layer in reversed(self.layers):
                        gradient = layer.backward_propagation(gradient, learning_rate, self.momentum, self.regularizator, self.nesterov)
                    outputs = []
                    targets = []
                    if not (lr_decay is None):
                        if (lr_decay == "linear"):
                            if learning_rate > lr_final:
                                lr_decay_alpha = i/lr_decay_finalstep
                                learning_rate = (1 - lr_decay_alpha) * initial_learning_rate + lr_decay_alpha * lr_final

            error /= N  # mean error over the set
            history.append(error)
            if not(val_history is None):
                # if a validation set is given, we now compute the error over it
                val_error = 0
                for j in range(M):
                    output = X_validation[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                    val_error += self.loss.forward(Y_validation[j], output)
                val_error /= M
                val_history.append(val_error)
                if not(metric is None):
                    metric_history.append(metric.compute(self, X_validation, Y_validation))
                if (verbose): training_progress(i+1, epochs, suffix=("loss = %f, val_loss = %f" % (error, val_error)))
            else:
                # if no validation set, we simply output the current status
                if (verbose): training_progress(i+1, epochs, suffix=("loss = %f" % (error)))
            if not(early_stopping is None):
                # with early stopping we need to check the current situation and
                # stop if needed
                if not(val_history is None):
                    # we use the validation error if a validation set is given
                    check_error = val_error
                else:
                    # otherwise we just use what we have, the training error
                    check_error = error
                if check_error >= min_error or np.isnan(check_error):
                    # the error is increasing or is stable, or there was an
                    # overflow situation, hence we are going toward an ES
                    es_epochs += 1
                    if es_epochs == early_stopping:
                        if not(val_history is None):
                            if (verbose): print('\nEarly stopping on epoch %d of %d with loss = %f and val_loss = %f' % (i+1, epochs, error, val_error))
                        else:
                            if (verbose): print('\nEarly stopping on epoch %d of %d with loss = %f' % (i+1, epochs, error))
                        break
                else:
                    # we're good
                    es_epochs = 0
                    min_error = check_error
        if (verbose): print("")

        # return the data that we have gathered
        if not(val_history is None):
            if not(metric is None):
                return history, val_history, metric_history
            else:
                return history, val_history
        else:
            if not(metric is None):
                return history, metric_history
            else:
                return history


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
