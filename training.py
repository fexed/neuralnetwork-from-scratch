import numpy as np

class Training():
    """ Base class for the neural networks training """

    def __init__(self, network, hyperparameters, logger):
    #momentum=0, dropout=0, nesterov=0, epochs=1000, learning_rate=0.01, early_stopping=None, batch_size=1, lr_decay=None, ):
        self.network = network
        self.hyperparameters = hyperparameters
        self.logger = logger


    def __call__(self, X_TR, Y_TR, X_VAl, Y_VAL, metric=None, verbose=True):
        return self.training_loop(X_TR, Y_TR, X_VAl, Y_VAL,  metric= metric, verbose=verbose, **self.hyperparameters)


    def training_epoch(self, X, Y, batch_size, eta): 
        batches_X = [X[i:i+batch_size] for i in range(0, len(X), batch_size)] 
        batches_Y = [Y[i:i+batch_size] for i in range(0, len(Y), batch_size)] 

        self.network.reset_gradients() #reset the gradient

        batch_gradient = 0
        for batch_X, batch_Y in zip(batches_X, batches_Y):
            for pattern, target in zip(batch_X, batch_Y):
                pattern_output = self.network.forward_propagation(pattern)
                batch_gradient += self.network.backward_propagation(pattern_output, target)

            self.network.update_weights(eta) # apply backprop and delta rule to update weights 


    def training_loop(self, X_TR, Y_TR, X_VAL=[], Y_VAL=[], metric=None, verbose=True,
        epochs=1000, learning_rate=0.01, early_stopping=None, batch_size=1, lr_decay=None):
 
        self.logger.training_preview()
        
        N = len(X_TR)
        tr_loss_hist = []
        val_loss_hist = []
        tr_metric_hist = []
        val_metric_hist = []

        self.es_epochs = 0  # counting early stopping epochs if needed
        self.min_error = float('inf') #@TODO Check why this is done
        self.early_stopping = early_stopping

        for i in range(epochs):
            # Training happens here
            self.training_epoch(X_TR, Y_TR, batch_size, learning_rate)

            # Should be checked!
            if not (lr_decay is None):
                if (lr_decay.type == "linear"):
                    if learning_rate > lr_decay.final_value:
                        lr_decay_alpha = i/lr_decay.last_step
                        learning_rate = (1 - lr_decay_alpha) * learning_rate + lr_decay_alpha * lr_decay.final_value

            # Compute learning curves 
            tr_output = self.network.forward_propagation(X_TR, inference=True)  
            #@TODO Should we calculate after weight update or reuse outputs from the forward_propagation propagation part?
            val_output = self.network.forward_propagation(X_VAL, inference=True)

            tr_loss_hist.append(self.network.loss.compute(tr_output, Y_TR))
            tr_metric_hist.append(metric.compute(tr_output, Y_TR))
            
            val_loss_hist.append(self.network.loss.compute(val_output, Y_VAL))
            val_metric_hist.append(metric.compute(val_output, Y_VAL))
    
            self.logger.training_progress(i, epochs, tr_loss_hist[i], val_loss_hist[i])

            if(self.early_stopping_condition(tr_loss_hist[i])):
                #self.network.logger.early_stopping_log()
                break
        
        return tr_loss_hist, val_loss_hist, tr_metric_hist, val_metric_hist 
    
    def early_stopping_condition(self, val): #@TODO Check Early Stopping Implementation
        if self.early_stopping:
            # with early stopping we need to check the current situation and
            # stop if needed
            check_error = val

            if check_error >= min_error or np.isnan(check_error):
                # the error is increasing or is stable, or there was an
                # overflow situation, hence we are going toward an ES
                es_epochs += 1
                if es_epochs == self.early_stopping:
                   return True
            else:
                # we're good
                es_epochs = 0
                min_error = check_error
                return False

