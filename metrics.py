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
                    val = 0 if out[i].item() < 0.5 else 1
                    if (targets[i].item() == val): accuracy += 1
            else:
                if (self.dataset == "CUP"):
                    if (compare(targets[i][0][0], out[i][0][0], tollerance = self.tollerance) and compare(targets[i][0][1], out[i][0][1], tollerance = self.tollerance)): accuracy += 1
                elif (self.dataset == "MONK"):
                    val = 0 if out[i].item() < 0.5 else 1
                    if (targets[i].item() == val): accuracy += 1
                else:
                    if (compare(targets[i][0], out[i][0][0], tollerance = self.tollerance)): accuracy += 1
        accuracy /= len(out)
        accuracy *= 100
        return accuracy
