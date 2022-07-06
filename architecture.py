from mlp import MLP

class Architecture():
    def __init__(self, model): 
        if model == MLP: 
            self.define = self.__init_MLP__
            return None

    def __init_MLP__(self, units, activations, loss, initializations='basic'):
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
