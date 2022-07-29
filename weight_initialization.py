import numpy as np
from math import sqrt
from hyperparameter import HyperParameter


class WeightInitialization(HyperParameter): 
    def __init__(self, name):
        super().__init__(f"{name} Weight Initialization", training=False)
        # self.input_size = input_size
        # self.output_size = output_size

    def generate(self, input_size, output_size): 
        #@TODO check if this can be deleted or should just be refactored in another subclass. 
        weights = np.random.rand(input_size, output_size) - 0.5  # so to have few <0 and few >0
        bias = np.random.rand(1, output_size) - 0.5  # so to have few <0 and few >0
        # return weights, bias
        pass

    def __str__(self):
        return self.name


class RandomUniform(WeightInitialization): 
    def __init__(self):
        super().__init__("Random Uniform")
    
    def generate(self, input_size, output_size):
        bias = [np.full(output_size, 0)]
        weights = np.random.uniform(-1/input_size, 1/input_size, (input_size, output_size))

        return weights, bias


class He(WeightInitialization): 
    def __init__(self):
        super().__init__("He")

    def generate(self, input_size, output_size):
        dev = sqrt(2.0 / input_size)
        weights = np.random.normal(loc=0.0, scale=dev, size=(input_size, output_size))
        bias = np.random.normal(loc=0.0, scale=dev, size=(1, output_size))

        return weights, bias


class Xavier(WeightInitialization): 
    def __init__(self):
        super().__init__("Xavier")

    def generate(self, input_size, output_size):
        l, u = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
        weights = np.random.uniform(low=l, high=u, size=(input_size, output_size))
        bias = np.random.uniform(low=l, high=u, size=(1, output_size))

        return weights, bias
        

class NormalizedXavier(WeightInitialization): 
    def __init__(self):
        super().__init__("Normalized Xavier")

    def generate(self, input_size, output_size):
        l, u = -(6.0 / sqrt(input_size + output_size)), (6.0 / sqrt(input_size + output_size))
        weights = np.random.uniform(low=l, high=u, size=(input_size, output_size))
        bias = np.random.uniform(low=l, high=u, size=(1, output_size))

        return weights, bias