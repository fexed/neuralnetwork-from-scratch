import numpy as np


class Network:
    def __init__(self, name="-unnamed-", loss=None, loss_deriv=None, regularizator=None, momentum=0):
        self.name = name
        self.layers = []
        self.loss = loss
        self.loss_deriv = loss_deriv
        self.regularizator = regularizator
        self.momentum = momentum


    def summary(self):
        trainable_parameters = 0
        print("Neural Network \"" + self.name + "\"")
        print("+==== Structure")
        for index, layer in enumerate(self.layers):
            print("|\t" + str(index+1) + ". " + type(layer).__name__, end = "")
            try:
                print(" of " +  str(len(layer.weights)) + " -> " + str(layer.weights[0].size) + " units", end = "")
                trainable_parameters += layer.weights.size
                trainable_parameters += layer.bias.size
            except AttributeError:
                pass
            try:
                print(" with " + layer.activation.__name__ + " activation function", end = "")
            except AttributeError:
                pass
            print("")
        print("+==== Loss: " + self.loss.__name__, end="")
        if not(self.regularizator is None):
            print(" and " + self.regularizator.__name__ + " regularizator", end="")
        if (self.momentum > 0):
            print(" and momentum = " + str(self.momentum), end="")
        print("")
        print("For a total of " + str(trainable_parameters) + " trainable parameters")


    def set_loss(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv


    def add(self, layer):  # TODO check input-output dimensions?
        self.layers.append(layer)


    def predict(self, data):
        N = len(data)
        results = []

        for i in range(N):
            output = data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            results.append(output)

        return results


    def training_loop(self, X, Y, X_validation=None, Y_validation=None, epochs=100, learning_rate=0.01, early_stopping=None, batch_size=None, verbose=True):
        N = len(X)
        history = []
        if not(X_validation is None):
            val_history = []
        else:
            val_history = None

        es_epochs = 0
        min_error = float('inf')
        for i in range(epochs):
            error = 0
            for j in range(N):
                if not(batch_size is None):
                    outputs = []
                    targets = []
                output = X[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                error += self.loss(Y[j], output)

                if not(batch_size is None):
                    outputs.append(output)
                    targets.append(Y[j])
                    if ((j+1) % batch_size == 0) or (j == N-1):
                        gradient = 0
                        for k in range(len(outputs)):
                            gradient += self.loss_deriv(targets[k], outputs[k])
                        gradient /= len(outputs)
                        for layer in reversed(self.layers):
                            gradient = layer.backward_propagation(gradient, learning_rate, self.momentum, self.regularizator)
                        outputs = []
                        targets = []
                else:
                    gradient = self.loss_deriv(Y[j], output)
                    for layer in reversed(self.layers):
                        gradient = layer.backward_propagation(gradient, learning_rate, self.momentum, self.regularizator)
            error /= N
            history.append(error)
            if not(val_history is None):
                M = len(X_validation)
                val_error = 0
                for j in range(M):
                    output = X[j]
                    for layer in self.layers:
                        output = layer.forward_propagation(output)
                    val_error += self.loss(Y_validation[j], output)
                val_error /= M
                val_history.append(val_error)
                if (verbose): print('Epoch %d of %d, loss = %f, val_loss = %f' % (i+1, epochs, error, val_error), end="\r")
            else:
                    if (verbose): print('Epoch %d of %d, loss = %f' % (i+1, epochs, error), end="\r")
            if not(early_stopping is None):
                if not(val_history is None):
                    check_error = val_error
                else:
                    check_error = error
                if check_error >= min_error:
                    es_epochs += 1
                    if es_epochs == early_stopping:
                        if not(val_history is None):
                            if (verbose): print('Early stopping on epoch %d of %d with loss = %f and val_loss = %f' % (i+1, epochs, error, val_error), end="\r")
                        else:
                            if (verbose): print('Early stopping on epoch %d of %d with loss = %f' % (i+1, epochs, error), end="\r")
                        break
                else:
                    es_epochs = 0
                    min_error = check_error
        if (verbose): print("")
        if not(val_history is None):
            return history, val_history
        else:
            return history
