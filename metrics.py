from utils import compare


def accuracy(model, dataset, targets, tollerance = 1e-03):
    out = model.predict(dataset)  # get the predictions
    accuracy = 0
    for i in range(len(out)):
        if (targets[0].shape[1] == 2):  # if CUP
            if (compare(targets[i][0][0], out[i][0][0], tollerance = tollerance) and compare(targets[i][0][1], out[i][0][1], tollerance = tollerance)): accuracy += 1
        else:  # else MONK
            val = 0 if out[i].item() < 0.5 else 1
            if (targets[i].item() == val): accuracy += 1
    accuracy /= len(out)
    accuracy *= 100
    return accuracy
