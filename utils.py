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


def tr_vl_split(X, Y, ratio=0.25, random_state=42):
    raise NotImplementedError


def plot(title, history, validation_history=None, ylabel="Loss", xlabel="Epochs", savefile=None):
    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(history)
    if not(validation_history is None): ax.plot(validation_history)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    if not(savefile is None): plot.savefig("plots/" + savefile + ".png")


def log(filename, data):
    import pickle
    with open("logs/"+ filename + ".pkl", "wb") as logfile:
        pickle.dump(data, logfile)
