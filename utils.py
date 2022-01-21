def update_progress(progress, barlength=100, prefix="", fill="\u2588"):
    """ Prints a progress bar with the current progress

    Parameters
    ----------
    progress : float
        The current progress, 0 <= progress <= 1
    barlength : int, optional
        The length of the progress bar
    prefix : str, optional
        A string to print before the progress bar
    fill : char, optional
        Character used to fill the progress bar
    """

    num = int(round(barlength*progress))
    txt = "\r" + prefix + " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
    print(txt, end="")


def training_progress(current_epoch, epochs, barlength=50, suffix="", fill="\u2588"):
    """ Prints a progress bar with the current training progress

    Parameters
    ----------
    current_epoch : int
        The current epoch of training
    epochs : int
        The max number of epochs of training
    barlength : int, optional
        The length of the progress bar
    prefix : str, optional
        A string to print before the progress bar
    fill : char, optional
        Character used to fill the progress bar
    """
    progress = current_epoch/epochs
    digits = len(str(epochs))
    formattedepochs = ("{:0"+str(digits)+"d}").format(current_epoch)
    num = int(round(barlength*progress))
    txt = "\rEpoch " + formattedepochs + " of " + str(epochs) + " " + suffix + " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
    print(txt, end="")


def tr_vl_split(X, Y, ratio=0.25):
    """ Splits a dataset into two parts with random elements

    Parameters
    ----------
    X
        The features to be splitted
    Y
        The targets to be splitted
    ratio : float, optional
        The ratio of the split
    """

    import numpy as np
    import math
    ix = np.random.randint(low = 0, high = len(X), size = math.floor(ratio * len(X)))
    X_vl, Y_vl = X[ix], Y[ix]
    X_tr, Y_tr = np.delete(X, ix, axis = 0), np.delete(Y, ix, axis = 0)
    return X_tr, X_vl, Y_tr, Y_vl


def plot_and_save(title, history, validation_history=None, ylabel="Loss", xlabel="Epochs", savefile=None):
    """ Plots some data and saves the image

    Parameters
    ----------
    title : str
        Title to be printed on top of the plot
    history : list
        The values to be printed
    validation_history : list, optional
        The values of the validation loss to be printed alongside the history
    ylabel : str, optional
        The label of the y axis
    xlabel : str, optional
        The label of the x axis
    savefile : str, optional
        The name of the file where to save the plot, in the plot folder
    """

    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(history, label=ylabel)
    if not(validation_history is None): ax.plot(validation_history, label='Validation Loss')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.legend()
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    if not(savefile is None): plot.savefig("plots/" + savefile + ".png")
    plot.clf()


def roc_curve(title, FPR, TPR, xlabel="Specificity", ylabel="Sensitivity", savefile=None):
    """ Plots the ROC curve

    Parameters
    ----------
    title : str
        Title to be printed on top of the plot
    FPR : list
        The false positive rate, x-axis
    TPR : list
        The true positive rate, y-axis
    xlabel : str, optional
        The label of the x axis
    ylabel : str, optional
        The label of the y axis
    savefile : str, optional
        The name of the file where to save the plot, in the plot folder
    """

    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.scatter(FPR, TPR)
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot.gca().margins(x=0)
    fig.set_size_inches(18.5, 10.5)
    if not(savefile is None): plot.savefig("plots/" + savefile + ".png")
    plot.clf()

def log(filename, data):
    """ Saves some data in a pickle file in the logs folder

    Parameters
    ----------
    filename : str
        The name of the file where to save
    data
        The data to be saved
    """

    import pickle
    with open("logs/"+ filename + ".pkl", "wb") as logfile:
        pickle.dump(data, logfile)


def compare(a, b, tollerance=1e-03):
    """ Compares two numbers with some tollerance

    Parameters
    ----------
    a : float
        The first number to be compared
    b : float
        The second number to be compared
    tollerance : float, optional
        The tollerance of the comparison

    Returns
    -------
    bool
        True if a and b are equal up to the tollerance
    """

    return abs(a - b) <= tollerance * max(abs(a), abs(b))


def shuffle(a,b):
    """ Shuffles two lists in parallel

    Parameters
    ----------
    a : list
        The first list to be shuffled
    b : list
        The second list to be shuffled

    Returns
    -------
    list, list
        The two lists, shuffled
    """

    import numpy as np
    assert len(a) == len(b)
    randomize = np.arange(len(a))
    np.random.shuffle(randomize)
    return a[randomize],  b[randomize]
