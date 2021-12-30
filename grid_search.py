import numpy as np
from activationfunctions import tanh, tanh_prime
from losses import MSE, MSE_prime
from layers import FullyConnectedLayer, ActivationLayer
from neuralnetwork import Network


def grid_search(input_size, output_size, X, y, X_validation=None, Y_validation=None, layers=list(range(1,5)), units=list(range(5, 100, 5)), learning_rates=list(np.arange(0.01, 0.1, 0.01)), batch_sizes=None, epochs=500, verbose=True):
    n_combinations = len(layers)*len(units)*len(learning_rates)
    if (verbose): print("Grid search on " + str(n_combinations) + " combinations")

    if (batch_sizes==None):
        batch_sizes=[None, input_size]
    results, parameters = [], []  # to store the results and return the best one
    for N in layers:
        for M in units:
            for E in learning_rates:
                for B in batch_sizes:
                    net = Network("GRIDSEARCH_" + str(N) + "L_" + str(M) + "U_" + str(E) + "LR", MSE, MSE_prime)
                    net.add(FullyConnectedLayer(input_size, M, tanh, tanh_prime))
                    for i in range(N):  # N -hidden- layers, plus input and output layers
                        net.add(FullyConnectedLayer(M, M, tanh, tanh_prime))
                    net.add(FullyConnectedLayer(M, output_size, tanh, tanh_prime))
                    if (verbose): net.summary()
    
                    if not(X_validation is None):
                        history, val_history = net.training_loop(X, y, X_validation=X_validation, Y_validation=Y_validation, epochs=epochs, learning_rate=E, batch_size=B, verbose=verbose)
                        results.append(history[-1])
                    else:
                        history = net.training_loop(X, y, epochs=epochs, learning_rate=E, batch_size=B, verbose=verbose)
                        results.append(history[-1])
    
                    parameters.append({"layers":N, "units":M, "learning_rate":E})

    results, parameters = zip(*sorted(zip(results, parameters)))  # sort both lists
    if (verbose): print("Best: " + "{:.5f}".format(results[0]) + " with " + str(parameters[0]))
    return parameters[0], results[0]
