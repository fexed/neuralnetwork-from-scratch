import itertools
import random

class HyperParameter(): 
    def __init__(self, name, training):
        self.name = name
        self.training = training


    def search_space(hp_type, range):
        return list(map(lambda r: hp_type(r), range))


class Epochs(HyperParameter):
    def __init__(self, n):
        super().__init__("Epochs", training = True) 
        self.key = 'epochs'
        self.n = n


    def value(self):
        return self.n


    def search_space(range): 
        return HyperParameter.search_space(Epochs, range)


    def __str__(self):
        return f"{self.name} number equal to {self.n}."  


class LearningRate(HyperParameter):
    def __init__(self, eta):
        super().__init__("Learning Rate", training = True)
        self.key = 'learning_rate'
        self.eta = eta


    def value(self):
        return self.eta


    def search_space(range): 
        return HyperParameter.search_space(LearningRate, range)


    def __str__(self):
        return f"{self.name} eta equal to {self.eta}."


class EarlyStopping(HyperParameter):
    def __init__(self, es):
        super().__init__("Early stopping", training = True)
        self.key = 'early_stopping'
        self.es = es


    def value(self):
        return self.es


    def search_space(range): 
        return HyperParameter.search_space(EarlyStopping, range)

 
    def __str__(self):
        return f"{self.name} since epoch {self.es}."


class BatchSize(HyperParameter): 
    def __init__(self, size):
        super().__init__("Batch Size", training = True) 
        self.key = 'batch_size'
        self.size = size


    def value(self):
        return self.size


    def search_space(range): 
        return HyperParameter.search_space(BatchSize, range)


    def __str__(self): 
        return f"{self.name} equal to {self.size}."


class LinearLearningRateDecay(HyperParameter): 
    def __init__(self, last_step=500, final_value=0.0001):
        super().__init__("Learning Rate Decay", training = True)
        self.key = 'lr_decay' 
        self.type = 'linear'
        self.last_step = last_step
        self.final_value = final_value


    def value(self):
        return self


    def search_space(last_step_range, final_value_range):
        range = itertools.product(last_step_range, final_value_range)
        return list(map(lambda r: LinearLearningRateDecay(r[0], r[1]), range))


    def __str__(self):
        return f"{self.name} from epoch {self.last_step} to value {self.final_value}."


class Momentum(HyperParameter): 
    def __init__(self, alpha=0):
        super().__init__("Momentum", training = False)
        self.key = 'momentum'
        self.alpha = alpha
        self.nesterov = False


    def value(self):
        return self


    def search_space(range): 
        return HyperParameter.search_space(Momentum, range)


    def __str__(self):
        return f"{self.name} with alpha coefficient equal to {self.alpha}."


class NesterovMomentum(Momentum): 
    def __init__(self, alpha=0):
        super().__init__(alpha)
        self.name = "Nesterov " + self.name
        self.nesterov = True


    def search_space(range): 
        return HyperParameter.search_space(NesterovMomentum, range)


class RandomizedMomentum(Momentum): 
    def __call__(self, alpha=0):
        return NesterovMomentum(alpha) if bool(random.getrandbits(1)) else Momentum(alpha)
        
        
    def search_space(range): 
        return HyperParameter.search_space(RandomizedMomentum, range)
        

class Dropout(HyperParameter): 
    def __init__(self, rate=1):
        super().__init__("Dropout", training = False)
        self.key = 'dropout'
        self.rate = rate


    def value(self):
        return self


    def search_space(range): 
        return HyperParameter.search_space(Dropout, range)


    def __str__(self): 
        return f"{self. name} rate equal to {self.rate}."
