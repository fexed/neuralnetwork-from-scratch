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

def multiline_plot(title, histories, legend_names, ylabel="Loss",  xlabel="Epochs", style="dark", showlegend=True, showgrid=False, savefile=None, alternateDots=False, prefix=""): 
    """ Plots multiple data curves on the same cartesian plane

    Parameters
    ----------
    title : str
        Title to be printed on top of the plot
    histories : list
        The list of cuves to be printed
    legend_names : list
        The list of legend names to be printed.  One for each history.
    ylabel : str, optional
        The label of the y axis
    xlabel : str, optional
        The label of the x axis
    style: string, optional
        Name of the seaborn style to be applied
    showlegend: boolean
        Show or hide legendnames
    showgrid: boolean
        Show or hide gridlines on the plot background. ]
    savefile : str, optional
        The name of the file where to save the plot, in the plot folder
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    l = len(histories)
    plt.rcParams.update({'font.size': 18})


    sns.set()

    with sns.color_palette(style, n_colors=l):
        fig, ax = plt.subplots()
        for  i  in reversed(range(l)):
            lStyle = '-' if ( alternateDots and i % 2 == 0) else ':'
            ax.plot(histories[i], label=legend_names[i], linestyle=lStyle, linewidth=2)
            
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_title(title)

        if (showlegend): ax.legend()
        if (showgrid): ax.grid(linestyle='--')
        
        plt.gca().margins(x=0)
        fig.set_size_inches(12, 8)
        if not(savefile is None): plt.savefig(prefix + "plots/" + savefile + ".png")
        plt.clf()

        log(prefix + "logs/" + savefile, histories)
    return

def roc_curve(title, FPR, TPR, AUC, xlabel="Specificity", ylabel="Sensitivity", savefile=None):
    """ Plots the ROC curve

    Parameters
    ----------
    title : str
        Title to be printed on top of the plot
    FPR : list
        The false positive rate, x-axis
    TPR : list
        The true positive rate, y-axis
    AUC : float
        The area under the ROC curve
    xlabel : str, optional
        The label of the x axis
    ylabel : str, optional
        The label of the y axis
    savefile : str, optional
        The name of the file where to save the plot, in the plot folder
    """

    import matplotlib.pyplot as plot
    fig, ax = plot.subplots()
    ax.plot(FPR, TPR)
    ax.plot([0, 1], [0, 1],'r--')
    ax.plot([], [], ' ', label="AUC = " + str(AUC))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plot.gca().margins(x=0)
    plot.xlim([-0.01,1.01])
    plot.ylim([-0.01,1.01])
    plot.legend()
    fig.set_size_inches(18.5, 10.5)
    if not(savefile is None): plot.savefig("plots/" + savefile + ".png")
    plot.clf()


# Only for MONK
def confusion_matrix(title="sample", values = (0,0,0,0), savefile=None):
    """ Plots the confusion matrix

    Parameters
    ----------
    title : str
        Title to be printed on top of the plot
    TP : int
        The true positives
    FN : int
        The false negatives
    FP : int
        The false positives
    TN : int
        The true negatives
    xlabel : str, optional
        The label of the x axis
    ylabel : str, optional
        The label of the y axis
    savefile : str, optional
        The name of the file where to save the plot, in the plot folder
    """

    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plot

    TP, TN, FP, FN = values

    cfm = [[TP,FP],
           [FN,TN]]

    x_labels = ["Predicted 1", "Predicted 2"]
    y_labels = ["Actually 1", "Actually 2" ]
    
    df_cm = pd.DataFrame(cfm, range(2), range(2))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True,  fmt="g", cmap="Blues" ,annot_kws={"size": 16}, xticklabels=x_labels, yticklabels=y_labels).set(title=title)

    plot.gca().margins(x=0)
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
    with open(filename + ".pkl", "wb") as logfile:
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