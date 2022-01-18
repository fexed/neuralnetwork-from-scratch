from utils import compare


def accuracy(model, dataset, targets, tollerance = 1e-03):
    out = model.predict(dataset)
    accuracy = 0
    for i in range(len(out)):
        # if (yvl[i][0][0] == out[i][0][0] and yvl[i][0][1] == out[i][0][1]): accuracy += 1
        if (compare(targets[i][0][0], out[i][0][0], tollerance = tollerance) and compare(targets[i][0][1], out[i][0][1], tollerance = tollerance)): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100
