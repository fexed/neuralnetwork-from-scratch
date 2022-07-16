class Logger(): 
    def __init__(self, verbose):
        self.set_verbosity(verbose)

    def set_verbosity(self, verbose):
        self.verobse = verbose
    
class GridSearchLogger(): 
    def __init__(self, grid_search): 
        self.grdi_search = grid_search

class MLPLogger(Logger): 
    def __init__(self, verbose=True):
        super().__init__(verbose)
    
    def summary(self, network, name): 
        trainable_parameters = 0  # output purposes
        print("Neural Network \"" + name + "\"")
        print("+==== Structure")
        nlayers = len(network.layers)
        try:
            print("|IN\t" + type(network.layers[0]).__name__ + ": " + str(len(network.layers[0].weights)) + " units" , end = "")
            print(" with " + network.layers[0].activation.name + " activation function", end = "")
        except AttributeError:
            pass
        print("")
        for i in range(nlayers-1):
            try:
                print("|HID\t" + type(network.layers[i]).__name__ + ": " + str(network.layers[i].weights[0].size) + " units" , end = "")
                print(" with " + network.layers[i].activation.name + " activation function", end = "")
            except AttributeError:
                pass
            print("")
        try:
            print("|OUT\t" + type(network.layers[-1]).__name__ + ": " + str(network.layers[-1].weights[0].size) + " units" , end = "")
            print(" with " + network.layers[-1].activation.name + " activation function", end = "")
        except AttributeError:
            pass
        print("")
        if not(network.loss is None):
            print("+==== Loss: " + network.loss.name, end="")
        else:
            print("+====")
        if not(network.regularization is None):
            print(" and " + network.regularization.name + " regularization with lambda = " + str(network.regularization.l), end="")
        if not(network.momentum is None):
            print(" and Momentum with alpha = ", network.momentum , end="")
        if (network.dropout.rate < 1):
            print(" and dropout = " + str(network.dropout.rate), end="")
        print("") 
        print("For a total of " + str(trainable_parameters) + " trainable parameters")

    def training_preview(self): 
        if (self.verobse):
            print("Beginetworking training loop with " + str(self.training.epochs) + " targeted epochs over " + str(self.training.N) + " training elements and learning rate = " + str(self.training.learning_rate), end="")
            if (self.training.batch_size > 1):
                print(" (batch size = " + str(self.training.batch_size) + ")", end="")
            if not(self.training.early_stopping is None):
                print(", with early stopping = " + str(self.training.early_stopping), end="")
            if len(self.training.X_VAL) != 0:
                print(" and validation set present", end="")
            if not(self.training.lr_decay is None):
                print(", with " + str(self.training.lr_decay))
            if not(self.training.metric is None):
                print(". The evaluation metric is " + self.training.metric.name, end="")
            print("")   
            

    def training_progress(self, current_epoch, epochs, tr_loss, val_loss, barlength=50, fill="\u2588"): 
        if (self.verobse): 
            progress = current_epoch/epochs
            digits = len(str(epochs))
            formattedepochs = ("{:0"+str(digits)+"d}").format(current_epoch)
            num = int(round(barlength*progress))

            losses = f"loss = {tr_loss}% " + f"val_loss = {val_loss}% " if not(val_loss is None) else "" 
            txt = "\rEpoch " + formattedepochs + " of " + str(epochs) + " " + losses
            bar = " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"

            print(txt + bar, end="")

    def early_stopping_log(self, i, tr_loss, val_loss):
        if (self.verobse): 
            print(f"\nEarly stopping on epoch {i+1} of {self.training.epochs} with loss={tr_loss }" +  
                    f"and val_loss = {val_loss}%f" if  not(val_loss is None) else "" )


class GridSearchLogger(Logger): 
    def __init__(self, search_space, verbose):
        super().__init__(verbose)

    def update_progress(self, progress, barlength=100, prefix="", fill="\u2588"):
        num = int(round(barlength*progress))
        txt = "\r" + prefix + " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
        print(txt, end="")