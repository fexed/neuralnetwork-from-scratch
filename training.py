import numpy as np
from utils import shuffle

class Training():
    """ Base class for the neural networks training """

    def __init__(self, network, hyperparameters, logger):
    #momentum=0, dropout=0, nesterov=0, epochs=1000, learning_rate=0.01, early_stopping=None, batch_size=1, lr_decay=None, ):
        self.network = network
        self.hyperparameters = hyperparameters
        self.logger = logger


    def __call__(self, X_TR, Y_TR, X_VAl, Y_VAL, metric=None, verbose=True, second_metric=None):
        return self.training_loop(X_TR, Y_TR, X_VAl, Y_VAL,  metric= metric, second_metric=second_metric, verbose=verbose, **self.hyperparameters)


    def training_epoch(self, X, Y, batch_size, eta, gradient_mean = True): 
        batches_X = [X[i:i+batch_size] for i in range(0, len(X), batch_size)] 
        batches_Y = [Y[i:i+batch_size] for i in range(0, len(Y), batch_size)] 

        self.network.reset_gradients() #reset the gradient

        for batch_X, batch_Y in zip(batches_X, batches_Y):
            for pattern, target in zip(batch_X, batch_Y):
                pattern_output = self.network.forward_propagation(pattern)
                self.network.backward_propagation(pattern_output, target)

            # Implicitly compute the mean over gradients in the same epoch, scaling the eta parameter.
            scaled_eta = eta 
            if gradient_mean: 
                scaled_eta *= len(batch_X)/len(X)

            self.network.update_weights(scaled_eta) # apply backprop and delta rule to update weights 


    def training_loop(self, X_TR, Y_TR, X_VAL=[], Y_VAL=[], metric=None, second_metric=None, verbose=True, epochs=1000, 
        learning_rate=0.01, early_stopping=None, batch_size=1, lr_decay=None, epoch_shuffle=True):
 
        self.logger.training_preview()
        
        second_metric_hist = [[], []]

        N = len(X_TR)
        tr_loss_hist = []
        val_loss_hist = []
        tr_metric_hist = []
        val_metric_hist = []

        self.es_epochs = 0  # counting early stopping epochs if needed
        self.min_error = float('inf') #@TODO Check why this is done
        self.early_stopping = early_stopping

        for i in range(epochs):
            # Compute learning curves 
            tr_output = self.network.forward_propagation(X_TR, inference=True)  
            #@TODO Should we calculate after weight update or reuse outputs from the forward_propagation propagation part?
            val_output = self.network.forward_propagation(X_VAL, inference=True)
            
            tr_loss_hist.append(self.network.loss.compute(tr_output, Y_TR))
            val_loss_hist.append(self.network.loss.compute(val_output, Y_VAL))
            
            tr_metric_hist.append(metric.compute(tr_output, Y_TR))
            val_metric_hist.append(metric.compute(val_output, Y_VAL))

            if second_metric:
                second_metric_hist[0].append(second_metric.compute(tr_output, Y_TR))
                second_metric_hist[1].append(second_metric.compute(val_output, Y_VAL))

            self.logger.training_progress(i, epochs, tr_loss_hist[i], val_loss_hist[i])

            if epoch_shuffle:
                X_TR,Y_TR = shuffle(X_TR, Y_TR)

            # Training happens here
            self.training_epoch(X_TR, Y_TR, batch_size, learning_rate)

            if not (lr_decay is None):
                if (lr_decay.type == "linear"):
                    if learning_rate > lr_decay.final_value:
                        lr_decay_alpha = i/lr_decay.last_step
                        learning_rate = (1 - lr_decay_alpha) * learning_rate + lr_decay_alpha * lr_decay.final_value



            if(self.early_stopping_condition(val_loss_hist[i])):
                #self.network.logger.early_stopping_log()
                break
        
        return [ tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist, *second_metric_hist ]
 
    
    def early_stopping_condition(self, val): #@TODO Check Early Stopping Implementation
        if self.early_stopping:
            # with early stopping we need to check the current situation and
            # stop if needed
            check_error = val

            if check_error >= self.min_error or np.isnan(check_error):
                # the error is increasing or is stable, or there was an
                # overflow situation, hence we are going toward an ES
                self.es_epochs += 1
                if self.es_epochs == self.early_stopping:
                   return True
            else:
                # we're good
                self.es_epochs = 0
                self.min_error = check_error
                return False

