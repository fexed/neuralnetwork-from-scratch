def update_progress(progress, barlength=100, prefix="", fill="\u2588"):
    num = int(round(barlength*progress))
    txt = "\r" + prefix + " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
    print(txt, end="")


def training_progress(current_epoch, epochs, barlength=50, suffix="", fill="\u2588"):
    progress = current_epoch/epochs
    digits = len(str(epochs))
    formattedepochs = ("{:0"+str(digits)+"d}").format(current_epoch)
    num = int(round(barlength*progress))
    txt = "\rEpoch " + formattedepochs + " of " + str(epochs) + " " + suffix + " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
    print(txt, end="")


def tr_vl_split(X, Y, ratio=0.25):
    import numpy as np
    import math
    ix = np.random.randint(low = 0, high = len(X), size = math.floor(ratio * len(X)))
    X_vl, Y_vl = X[ix], Y[ix]
    X_tr, Y_tr = np.delete(X, ix, axis = 0), np.delete(Y, ix, axis = 0)
    return X_tr, X_vl, Y_tr, Y_vl


def plot_loss(title, history, validation_history=None, ylabel="Loss", xlabel="Epochs", savefile=None):
    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(history, label='Loss')
    if not(validation_history is None): ax.plot(validation_history, label='Validation Loss')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    if not(savefile is None): plot.savefig("plots/" + savefile + ".png")
    plot.clf()


def log(filename, data):
    import pickle
    with open("logs/"+ filename + ".pkl", "wb") as logfile:
        pickle.dump(data, logfile)


def compare(a, b, tollerance=1e-03):
    return abs(a - b) <= tollerance * max(abs(a), abs(b))
