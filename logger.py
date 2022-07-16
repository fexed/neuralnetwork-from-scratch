class Logger(): 
    def __init__(self, verbose):
        self.set_verbosity(verbose)


    def set_verbosity(self, verbose):
        self.verobse = verbose
    

class GridSearchLogger(Logger): 
    def __init__(self, grid_search): 
        self.grdi_search = grid_search


class MLPLogger(Logger): 
    def __init__(self, name, architecture, hyperparameters, verbose=True):
        self.name = name
        self.architecture = architecture
        self.hyperparameters = hyperparameters
        super().__init__(verbose)
    

    def summary(self): 
        print("Neural Network \"" + self.name + "\"")
        print(self.architecture)
        
        for hp in self.hyperparameters: 
            print(hp)


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