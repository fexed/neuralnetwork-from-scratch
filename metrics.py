class Metric():
    def __init__(self):
        self.name = None


class Accuracy(Metric):
    def __init__(self, dataset=None, tollerance=1e-03):
        self.name = "Accuracy"
        self.dataset = dataset
        self.tollerance = tollerance


    def compute(self, model, dataset, targets):
        from utils import compare
        out = model.predict(dataset)  # get the predictions
        accuracy = 0
        for i in range(len(out)):
            if (self.dataset is None):
                if len(targets[0].shape) == 1:  # if just one target
                    if (compare(targets[i][0], out[i][0][0], tollerance = self.tollerance)): accuracy += 1
                elif (targets[0].shape[1] == 2):  # if normal CUP
                    if (compare(targets[i][0][0], out[i][0][0], tollerance = self.tollerance) and compare(targets[i][0][1], out[i][0][1], tollerance = self.tollerance)): accuracy += 1
                else:  # else MONK
                    accuracy += (1-abs(out[i].item() - targets[i].item()))
            else:
                if (self.dataset == "CUP"):
                    if (compare(targets[i][0][0], out[i][0][0], tollerance = self.tollerance) and compare(targets[i][0][1], out[i][0][1], tollerance = self.tollerance)): accuracy += 1
                elif (self.dataset == "MONK"):
                    accuracy += (1-abs(out[i].item() - targets[i].item()))
                else:
                    if (compare(targets[i][0], out[i][0][0], tollerance = self.tollerance)): accuracy += 1
        accuracy /= len(out)
        accuracy *= 100
        return accuracy


class MeanEuclideanError(Metric):
    def __init__(self):
        self.name = "Mean Euclidean Error"

    def compute(self, model, dataset, targets):
        import numpy as np
        predictions = model.predict(dataset)
        MEE = (np.linalg.norm(predictions - targets)/len(targets))
        return MEE


class ConfusionMatrix(Metric):
    def __init__(self, threshold=0.5):
        self.name = "Confusion Matrix"
        self.threshold = threshold


    def compute(self, model, dataset, targets):
        predictions = model.predict(dataset)

        # 1 is positive, 0 is negative
        TP, TN, FP, FN = 0, 0, 0, 0

        for predicted, target in zip(predictions, targets):
            predicted = 0 if predicted.item() < self.threshold else 1
            target = target.item()
            if (target == 1):
                if (predicted == 1): TP += 1
                else: FN += 1
            else:
                if (predicted == 1): FP += 1
                else: TN += 1

        specificity = TN/(FP + TN + 1e-5)
        sensitivity = TP/(TP + FN + 1e-5)
        precision = TP/(TP + FP + 1e-5)
        accuracy = (TP + TN)/len(targets)

        return accuracy, specificity, sensitivity, precision, (TP, TN, FP, FN)


class ROCCurve(Metric):
    def __init__(self, thresholds=None):
        self.name = "ROC Curve"
        if not(thresholds is None): self.thresholds = thresholds
        else:
            self.thresholds = []
            for i in range(1000): self.thresholds.append(i/1000)


    def compute(self, model, dataset, targets):
        predictions = model.predict(dataset)

        FPR, TPR = [], []
        for threshold in self.thresholds:
            TP, TN, FP, FN = 0, 0, 0, 0

            for predicted, target in zip(predictions, targets):
                predicted = 0 if predicted.item() < threshold else 1
                target = target.item()
                if (target == 1):
                    if (predicted == 1): TP += 1
                    else: FN += 1
                else:
                    if (predicted == 1): FP += 1
                    else: TN += 1

            n = FP/(FP + TN)
            m = TP/(TP + FN)

            FPR.append(n)
            TPR.append(m)
        AUC = 0
        for i in range(len(self.thresholds)-1):
            AUC += (FPR[i] - FPR[i+1]) * TPR[i]
        return FPR, TPR, AUC
