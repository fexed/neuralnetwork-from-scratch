import itertools
from math import comb
from activationfunctions import Identity
from mlp import MLP

class Architecture():
    def __init__(self, model): 
        if model == MLP: 
            self.define = self.__init_MLP__
            self.search_space = self.__search_space_MLP__


    def __init_MLP__(self, loss, units, activations,  initializations=['basic']):
        self.units = units
        self.activations = activations
        self.loss = loss

        layer_num = len(self.units) - 1

        if len(list(activations)) == 1: 
            self.activations = list(activations) * layer_num
        elif len(activations) == len(self.units) - 1:
            self.activations = activations
        else: 
            raise Exception("Activation functions do not match with the number layers")

        if len(list(initializations)) == 1: 
            self.initializations = list(initializations) * layer_num
        elif len(initializations) == len(self.units) - 1:
            self.initializations = initializations
        else: 
            raise Exception("Initialization functions do not match with the number layers")

        return self

    def __search_space_MLP__(self, io_sizes, loss,  hidden_units, activation, initialization, last_activation=None):
        combinations =  itertools.product(hidden_units,  activation, initialization)

        archs = []
        for c in combinations:
            archs.append(Architecture(MLP).define(
                loss=loss,
                units = [io_sizes[0], *c[0], io_sizes[1] ], #*hidden_units, 
                activations = [c[1]], 
                initializations = [c[2]],
            ))

        if not (last_activation is None):
            for a in archs:
                a.activations[-1] = last_activation

        return archs

    def __str__(self) -> str:
        return f"Architecture composed by {len(self.units)} layers of { self.units } units. Objective function is {self.loss}. Last layer has {self.activations[-1]}" 