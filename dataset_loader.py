from preprocessing import one_hot_encoding
import numpy as np

MONK_INPUT_SIZE=6

def load_monk(monk=1, test=False, use_one_hot=False, verbose=True ):
    if (verbose):
        print("\n\n****MONK" + str(monk))

    monkfile = open("datasets/MONK/monks-" + str(monk) + (".test"  if test else ".train"), "r")
    xtr = []
    ytr = []
    for line in monkfile.readlines():
        vals = line.split(" ")

        xtr.append([[int(vals[2]), int(vals[3]), int(vals[4]), int(vals[5]), int(vals[6]), int(vals[7])]])
        ytr.append([[int(vals[1])]])
    X = np.array(xtr)
    Y = np.array(ytr)

    input_size = MONK_INPUT_SIZE

    if use_one_hot:
        X, input_size = one_hot_encoding(X)

    return X,Y, input_size

def load_cup(verbose=True, test=False, forfirst=False):
    if (verbose): ("\n\n****CUP")
    cupfile = open("datasets/CUP/ML-CUP21-" + ("TS" if test else "TR") + ".csv", "r")
    xtr = []
    ytr = []
    for line in cupfile.readlines():
        if (line.startswith("#")):
            continue
        vals = line.split(",")
        if (forfirst):
            xtr.append([[float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7]), float(vals[8]), float(vals[9]), float(vals[10]), float(vals[11])]])
        else:
            xtr.append([[float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7]), float(vals[8]), float(vals[9]), float(vals[10])]])
        if not test:
            if (forfirst):
                ytr.append([[float(vals[12])]])
            else:
                ytr.append([[float(vals[11]), float(vals[12])]])

    X = np.array(xtr)

    return X if test else X, np.array(ytr)
