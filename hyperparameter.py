class HyperParameter(): 
    def __init__(self, name, training):
        self.name = name
        self.training = training

    def value(self):
        pass

    def range(self):
        return None
    
    def __str__(self):
        return f"{self.name}"  


class Epochs(HyperParameter):
    def __init__(self, n):
        super().__init__("Epochs", training = True) 
        self.key = 'epochs'
        self.n = n

    def value(self):
        return self.n


class LearningRate(HyperParameter):
    def __init__(self, eta):
        super().__init__("Learning Rate", training = True)
        self.key = 'learning_rate'
        self.eta = eta

    def value(self):
        return self.eta


class EarlyStopping(HyperParameter):
    def __init__(self, es):
        super().__init__("Early stopping", training = True)
        self.key = 'early_stopping'
        super.es = es

    def value(self):
        return self.es


class BatchSize(HyperParameter): 
    def __init__(self, size):
        super().__init__("Batch Size", training = True) 
        self.key = 'batch_size'
        self.size = size

    def value(self):
        return self.size


class LinearLearningRateDecay(HyperParameter): 
    def __init__(self, last_step=500, final_value=0.0001):
        super().__init__("Learning Rate Decay", training = True)
        self.key = 'lr_decay' 
        self.type = 'linear'
        self.last_step = last_step
        self.final_value = final_value

    def value(self):
        return self


class Momentum(HyperParameter): 
    def __init__(self, alpha=0):
        super().__init__("Momentum", training = False)
        self.key = 'momentum'
        self.alpha = alpha
        self.nesterov = False

    def value(self):
        return { self.alpha, self.nesterov }


class NesterovMomentum(Momentum): 
    def __init__(self, alpha=0):
        super().__init__(alpha)
        self.name = "Nesterov " + self.name
        self.nesterov = True


class Dropout(HyperParameter): 
    def __init__(self, rate=1):
        super().__init__("Dropout", training = False)
        self.rate = rate

    def value(self):
        return self.rate
