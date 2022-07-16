import itertools
from mlp import MLP

class Architecture():
    def __init__(self, model): 
        if model == MLP: 
            self.define = self.__init_MLP__
            self.search_space = self.__search_space_MLP__
            self.__str__ = self.__str_MLP__


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


    def __str_MLP__(self) -> str:
        hidden_layers = ""

        for i in range(0, len(self.units) - 1):
            hidden_layers += f"|HID\t Hidden layer with {self.units[i+1]} and {self.activations[i]} - {self.initializations[i]}"

        return f""" 
                |IN\t Input layer with {self.units[0]} units - 
                {hidden_layers}
                |OUT\t Output layer with {self.units[-1]} units and {self.activations[-1]} - {self.initializations[-1]}
                """
    