class Logger(): 
    def __init__(self, verbose):
        self.set_verbosity(verbose)


    def set_verbosity(self, verbose):
        self.verbose = verbose
    

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
            if not hp.training: print(hp)


    def training_preview(self): 
        print("")
        if self.verbose:
            print("Begin network training loop with:")
            for hp in self.hyperparameters: 
                if hp.training: print(hp)
            

    def training_progress(self, current_epoch, epochs, tr_loss, val_loss, barlength=50, fill="\u2588"): 
        if (self.verbose): 
            progress = current_epoch/epochs
            digits = len(str(epochs))
            formattedepochs = ("{:0"+str(digits)+"d}").format(current_epoch)
            num = int(round(barlength*progress))

            losses = f"loss = {tr_loss}% " + f"val_loss = {val_loss}% " if not(val_loss is None) else "" 
            txt = "\rEpoch " + formattedepochs + " of " + str(epochs) + " " + losses
            bar = " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"

            print(txt + bar, end="")


    def early_stopping_log(self, i, tr_loss, val_loss):
        if (self.verbose): 
            print(f"\nEarly stopping on epoch {i+1} of {self.training.epochs} with loss={tr_loss }" +  
                    f"and val_loss = {val_loss}%f" if  not(val_loss is None) else "" )


    def __print_results(self, title, loss_val, metric_val, metric): 
        if self.verbose:     
            print("")
            print(f"+==== {title} results ({self.name}): =====  ===== ====+")
            print(f"+\t- {self.architecture.loss} = {loss_val}")
            print(f"+\t- Evaluated {metric} = {metric_val}")
            print("+==== ===== ==== ===== ==== ==== ===== ==== ====+")


    def training_results(self, loss_val, metric_val, metric):
        self.__print_results("Training", loss_val, metric_val, metric)


    def validation_results(self, loss_val, metric_val, metric):
        self.__print_results("Validation", loss_val, metric_val, metric)


    def test_results(self, loss_val, metric_val, metric):
        self.__print_results("Test", loss_val, metric_val, metric)


class GridSearchLogger(Logger): 
    def __init__(self, search_name, dataset_name, dataset_size, model_type, model_number, verbose):
        self.search_name = search_name
        self.dataset_name = dataset_name
        self.dataset_size = dataset_size
        self.model_type = model_type
        self.model_number = model_number
        super().__init__(verbose)


    def search_preview(self, folding_strategy):
        if self.verbose: 
            print(f"+==== Grid Search: \" {self.search_name}\" ==== ==== ==== ====+")
            print(f"+-\t Target dataset is: {self.dataset_name}, composed by {self.dataset_size} instances.")
            print(f"+-\t {folding_strategy}")
            print(f"+-\t Model is {self.model_type}.")
            print(f"+-\t Architecture & Hyperparameter combinations: {self.model_number}.")
            print(f"+-\t Training cycles: {self.model_number*folding_strategy.k}.")
            print(f"+==== ==== ==== ==== ==== ==== ===`= ===== ===== ==== =====+")


    def update_progress(self, progress, barlength=100, prefix="", fill="\u2588"):
        num = int(round(barlength*progress))
        txt = "\r" + prefix + " [" + fill*num + " "*(barlength - num) + "] " + "{:.2f}".format(progress*100) + "%"
        print(txt, end="")

    def end_message(self):
        print("")
        print("GridSearch succefully terminated!")
        print("")

    def top_result_line(self, i, j, mean, std): 
        line =  f"{i}): Model: {j}_* --- Mean: {mean} (Standard deviation over folds: {std})"

        if self.verbose: 
            print(line)

        return line
