from enum import Enum
from losses import MEE, MSE
import numpy as np

class Task(Enum): 
    REGRESSION = 0
    BINARY_CLASSIFICATION = 1,
    # MULTICLASS_CLASSIFICATION = 2


class Metric():
    def __init__(self, name: str, task: Task, is_loss = False):
        self.name = name
        self.task = task
        self.is_loss = is_loss


    def compute(self, output, target): 
        return None


    def __str__(self): 
        return f"Metric: {self.name}"


class Accuracy(Metric):
    def __init__(self):
        super().__init__("Accuracy", Task.BINARY_CLASSIFICATION)


    def compute(self, output, target):
        TP, TN, _, _  = logistic_to_confusion_matrix(output, target)
        return  (TP + TN) / len(target)


class Precision(Metric): 
    def __init__(self): 
        super().__init__("Precision", Task.BINARY_CLASSIFICATION)


    def compute(self, output, target):
        TP, _, FP, _  = logistic_to_confusion_matrix(output, target)
        return  TP/(TP + FP + 1e-5)


class Recall(Metric): 
    def __init__(self): 
        super().__init__("Recall (Sensitivity)", Task.BINARY_CLASSIFICATION)


    def compute(self, output, target):
        TP, _, _, FN  = logistic_to_confusion_matrix(output, target)
        return  TP/(TP + FN + 1e-5)


class Specificity(Metric): 
    def __init__(self): 
        super().__init__("Specificity", Task.BINARY_CLASSIFICATION)


    def compute(self, output, target):
        _, TN, FP, _  = logistic_to_confusion_matrix(output, target)
        return  TN/(FP + TN + 1e-5)


class MeanSquaredError(Metric): 
    def __init__(self): 
        super().__init__("Mean Squared Error", Task.REGRESSION, True)


    def compute(self, output, target): 
        # This MUST be fixed when refactoring training loop 
        return MSE().compute(output, target)


class MeanEuclideanError(Metric):
    def __init__(self):
        super().__init__("Mean Euclidean Error", Task.REGRESSION, True)


    def compute(self, output, target):      
        # This MUST be fixed when refactoring training loop 
        return MEE().compute(output, target)


def logistic_to_confusion_matrix(output, target):
    #if x.shape[1] != 1:
        #raise Exception("Multinomial classification not supported yet")
    
    TP, TN, FP, FN = 0, 0, 0, 0

    #  0 is negative, 1 is positive.
    for x_i, t_i in zip(output, target):
        x_i = 0 if x_i[0] < 0.5 else 1
        if (t_i == 1):
            if (x_i == 1): TP += 1
            else: FN += 1
        else:
            if (x_i == 1): FP += 1
            else: TN += 1

    return TP, TN, FP, FN