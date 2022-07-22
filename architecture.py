import itertools
from mlp import MLP
from search_space import SearchSpace
from weight_initialization import RandomUniform

class Architecture():
    def __init__(self, model): 
        self.model = model
        if model == MLP: 
            self.define = self.__init_MLP__
            self.search_space = self.__search_space_MLP__
            self.__str__ = self.__str_MLP__


    def __init_MLP__(self, loss, units, activations, initializations=[RandomUniform()]):
        self.units = units
        self.loss = loss

        layer_num = len(self.units) - 1

        if len(activations) == 1: 
            self.activations = list(activations) * layer_num
        elif len(activations) == len(self.units) - 1:
            self.activations = activations
        else: 
            raise Exception("Activation functions do not match with the number layers")

        print(len(self.activations))

        if len(initializations) == 1: 
            self.initializations = list(initializations) * layer_num
        elif len(initializations) == len(self.units) - 1:
            self.initializations = initializations
        else: 
            raise Exception("Initialization functions do not match with the number layers")

        return self


    def __search_space_MLP__(self, io_sizes, loss,  hidden_units, activation, initialization, last_activation=None):
        combinations =  itertools.product(hidden_units,  activation, initialization)
        #@TODO RIPRENDERE DA QUIII.
        archs = []
        for c in combinations:
            archs.append(Architecture(MLP).define(
                loss=loss,
                units = [io_sizes[0], *c[0], io_sizes[1]], #*hidden_units, 
                activations = c[1], 
                initializations = c[2],
            ))

        if not (last_activation is None):
            for a in archs:
                a.activations[-1] = last_activation

        return archs


    def __str__(self): 
        if self.model == MLP: 
            return self.__str_MLP__()


    def __str_MLP__(self) -> str:
        
        str = f"+==== Architecture ====+ \n"
        str += f"|IN\t Input layer with {self.units[0]} units. \n"
        for i in range(0, len(self.units) - 2):
            str += f"|HID\t Hidden layer with {self.units[i+1]} units and {self.activations[i]} - {self.initializations[i]}. \n"
    

        str += f"|OUT\t Output layer with {self.units[-1]} units and {self.activations[-1]} - {self.initializations[-1]}. \n"
        str += f"+==== {self.loss} ====+"
        return str

    