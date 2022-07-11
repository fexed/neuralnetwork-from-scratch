from hyperparameter import HyperParameter
from logger import MLPLogger
from model import Model
from layers import FullyConnectedLayer
from training import Training
from utils import multiline_plot, log



class MLP(Model): 
    def __init__(self, name, architecture, hyperparameters = []):        
        self.loss = architecture.loss 
        self.units = architecture.units

        self.activations = architecture.activations
        self.initializations = architecture.initializations

        self.layers = []

        for i in range(len(self.units) - 1):
            self.layers.append(FullyConnectedLayer(self.units[i], self.units[i+1], self.activations[i], self.initializations[i]))
        
        self.structural_hps = {} 
        self.training_hps = {}

        for hp in hyperparameters: 
            self.set_hyperparameter(hp)
        
        self.trained = False
        self.training_algorithm = Training(self, self.training_hps)

        self.logger = MLPLogger(self, True)

        super().__init__(name + '_MLP')
        #self.logger.summary() #Check parmaeters passage


    def set_hyperparameter(self, hp: HyperParameter):
        getattr(self, 'training_hps' if hp.training else 'structural_hps')[hp.key] = hp.value()
    
    
    def reset(self): 
        print(f"Resetting the {self.name}")
        self.build()


    def predict(self, X):
        output = []

        for i in range(len(X)):
            o = X[i]
            for layer in self.layers:
                o = layer.forward_propagation(o)
            output.append(o)

        return output

    def forward_propagation(self, p):
        """ Performs the forward propagation of the network """
        output = p
        for layer in self.layers:
            output = layer.forward_propagation(output, dropout=1)

        return output

    def backward_propagation(self, output, target):
        """ Performs the backward propagation of the network """

        gradient = self.loss.derivative(output, target)

        for layer in reversed(self.layers):
            gradient = layer.backward_propagation(gradient)
        
        return gradient


    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()


    def update_weights(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate, **self.structural_hps)


    def add(self, layer):
        """ Adds another layer at the bottom of the network """
        self.layers.append(layer)


    def train(self, X_TR, Y_TR, X_VAL, Y_VAL, metric, verbose = True, plot_folder=''): 
        self.metric = metric #Not so elegant... 

        tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist = self.training_algorithm(
            X_TR, Y_TR, X_VAL, Y_VAL, metric=metric, verbose=verbose
        )

        history = [tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist]
        
        log(f'{self.path}logs/training', history)

        self.plot_training_curves(history[0:2], self.loss.name, plot_folder)
        self.plot_training_curves(history[2:4], metric.name, plot_folder)

        self.tr_loss = tr_loss_hist[-1]
        self.val_loss = val_loss_hist[-1]
        
        self.tr_metric = tr_metric_hist[-1]
        self.val_metric = val_metric_hist[-1]
        
        self.trained = True

        
    def plot_training_curves(self, history, metric_name, folder ):
        legend_names = ["TR", "VL"] if len(history) == 2 else ["TR"]
            
        multiline_plot(
            title = f"{self.name}_{metric_name}",
            legend_names = legend_names,
            histories=list(history),
            ylabel=metric_name, xlabel="Epochs", 
            showlegend=True, showgrid=True, alternateDots=True,
            savefile=f"{metric_name}_TR", prefix = folder + self.path
        )


    def save(self):
        """ Saves the neural network in a pickle """
        super().save('MLP')
