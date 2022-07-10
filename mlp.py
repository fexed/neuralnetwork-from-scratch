from model import Model
from layers import FullyConnectedLayer
from neuralnetwork import Network
from utils import multiline_plot, log


class MLP(Model): 
    def __init__(self, name, architecture, hyperparameters = []):
        super().__init__(name + '_MLP')
        
        self.loss = architecture.loss 
        self.units = architecture.units

        self.activations = architecture.activations
        self.initializations = architecture.initializations

        self.training_hps = {}
        self.structural_hps = {} 

        for hp in hyperparameters: 
            getattr(self, 'training_hps' if hp.training else 'structural_hps')[hp.key] = hp.value()

        self.build()


    def build(self): 
        reg = self.structural_hps.get('reg')
        net = Network(self.name, self.loss, regularizator = reg)

        for i in range(len(self.units) - 1):
            net.add(FullyConnectedLayer(self.units[i], self.units[i+1], self.activations[i], self.initializations[i]))
        
        net.summary()
        self.net = net

        self.trained = False
        self.evaluated = False

        super().create_model_folder()
    

    def reset(self): 
        print(f"Resetting the {self.name}")
        self.build()


    def train(self, X_TR, Y_TR, X_VAL, Y_VAL, metric, verbose = True, plot_folder=''): 
        self.metric = metric #Not so elegant... 

        tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist = self.net.training_loop(
            X_TR, Y_TR, X_VAL, Y_VAL, **self.training_hps, metric=metric, verbose=verbose
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
    
    
    def save(self):
        # Probably this is not enough now
        self.net.savenet(f'{self.path}logs/mlp.pkl')


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


    def predict(self, X): 
        return self.net.predict(X)


    def evaluate(self, X_TS, Y_TS, metric = None): 
        if metric and self.metric:
            self.metric = metric

        output = self.predict(X_TS)

        self.ts_loss = self.loss.compute(output, Y_TS)
        self.ts_metric = self.metric.compute(output, Y_TS)

        self.evaluated = True
        self.save()
    
    def results(self):
        print(f"TR_LOSS:_{self.tr_loss}")
        print(f"VAL_LOSS:_{self.val_loss}")
        
        print(f"TR_METRIC_{self.tr_metric}")
        print(f"VAL_METRIC:_{self.val_metric}")

        if self.evaluated:
            print(f"TS_LOSS:_{self.ts_loss}")
            print(f"TS_METRIC:_{self.ts_metric}")