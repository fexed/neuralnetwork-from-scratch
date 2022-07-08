from enum import Enum
from losses import MEE, MSE

class Task(Enum): 
    REGRESSION = 0
    BINARY_CLASSIFICATION = 1,
    # MULTICLASS_CLASSIFICATION = 2


class Metric():
    def __init__(self, name: str, task: Task):
        self.name = name
        self.task = task

    def compute(self, x, y): 
        return None


class Accuracy(Metric):
    def __init__(self):
        super().__init__("Accuracy", Task.BINARY_CLASSIFICATION)

    def compute(self, x, y):
        TP, TN, _, _  = logistic_to_confusion_matrix(x, y)
        return  (TP + TN) / len(y)


class Precision(Metric): 
    def __init__(self): 
        super().__init__("Precision", Task.BINARY_CLASSIFICATION)

    def compute(self, x, y):
        TP, _, FP, _  = logistic_to_confusion_matrix(x, y)
        return  TP/(TP + FP + 1e-5)


class Recall(Metric): 
    def __init__(self): 
        super().__init__("Recall (Sensitivity)", Task.BINARY_CLASSIFICATION)

    def compute(self, x, y):
        TP, _, _, FN  = logistic_to_confusion_matrix(x, y)
        return  TP/(TP + FN + 1e-5)


class Specificity(Metric): 
    def __init__(self): 
        super().__init__("Specificity", Task.BINARY_CLASSIFICATION)

    def compute(self, x, y):
        _, TN, FP, _  = logistic_to_confusion_matrix(x, y)
        return  TN/(FP + TN + 1e-5)


class MeanSquaredError(): 
    def __init__(self): 
        super().__init__("Mean Squared Error", Task.REGRESSION)

    def compute(self, x, target): 
        return MSE().forward(x, target)


class MeanEuclideanError(Metric):
    def __init__(self):
        super().__init__("Mean Euclidean Error", Task.REGRESSION)

    def compute(self, x, target): 
        return MEE().forward(x, target)


def logistic_to_confusion_matrix(x, target):
    #if x.shape[1] != 1:
        #raise Exception("Multinomial classification not supported yet")
    
    TP, TN, FP, FN = 0, 0, 0, 0

    #  0 is negative, 1 is positive.
    for x_i, t_i in zip(x, target):
        x_i = 0 if x_i[0] < 0.5 else 1
        if (t_i == 1):
            if (x_i == 1): TP += 1
            else: FN += 1
        else:
            if (x_i == 1): FP += 1
            else: TN += 1

    return TP, TN, FP, FN


# class ROCCurve(Metric):
#     def __init__(self, thresholds=None):
#         self.name = "ROC Curve"
#         if not(thresholds is None): self.thresholds = thresholds
#         else:
#             self.thresholds = []
#             for i in range(1000): self.thresholds.append(i/1000)


#     def compute(self, model, dataset, targets):
#         predictions = model.predict(dataset)

#         FPR, TPR = [], []
#         for threshold in self.thresholds:
#             TP, TN, FP, FN = 0, 0, 0, 0

#             for predicted, target in zip(predictions, targets):
#                 predicted = 0 if predicted.item() < threshold else 1
#                 target = target.item()
#                 if (target == 1):
#                     if (predicted == 1): TP += 1
#                     else: FN += 1
#                 else:
#                     if (predicted == 1): FP += 1
#                     else: TN += 1

#             n = FP/(FP + TN)
#             m = TP/(TP + FN)

#             FPR.append(n)
#             TPR.append(m)
#         AUC = 0
#         for i in range(len(self.thresholds)-1):
#             AUC += (FPR[i] - FPR[i+1]) * TPR[i]
#         return FPR, TPR, AUC

# class ConfusionMatrix(Metric):
#     def __init__(self, threshold=0.5):
#         self.name = "Confusion Matrix"
#         self.threshold = threshold