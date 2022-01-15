import numpy as np
from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from regularizators import L2
from utils import update_progress
import time


def grid_search(input_size, output_size, X, y,
                X_validation=None, Y_validation=None,
                layers=list(range(5)),  # number of -hidden- layers
                units=list(range(5, 100, 5)),
                learning_rates=list(np.arange(0.01, 0.1, 0.01)),
                batch_sizes=None,
                init_functions=["xavier", "normalized_xavier", "he", "basic"],
                momentums=[0, 0.8, 0.9, 0.99, 0.999],
                regularizators=[None, "L2"],
                regularization_lambdas=[0.005, 0.01, 0.15],
                epochs=500, verbose=True, early_stopping=25):

    if (batch_sizes==None):
        batch_sizes=[1, input_size]

    n_combinations = len(layers)*len(units)*len(learning_rates)*len(init_functions)*len(momentums)*len(regularizators)*len(regularization_lambdas)*len(batch_sizes)
    if (verbose): print("Grid search on " + str(n_combinations) + " combinations")
    results = []  # to store the results and return the best 10

    try:
        tested = 0  # for the progressbar
        start = time.time()  # for ETA calc
        for N in layers:
            for M in units:
                for E in learning_rates:
                    for momentum in momentums:
                        for regularization_lambda in regularization_lambdas:
                            for regularizatorname in regularizators:
                                if regularizatorname == "L2": regularizator = L2
                                else: regularizator = None
                                for init_f in init_functions:
                                    for B in batch_sizes:
                                        net = Network("GRIDSEARCH_" + str(N) + "L_" + str(M) + "U_" + str(E) + "LR", binary_crossentropy, binary_crossentropy_prime, momentum=momentum, regularizator=regularizator, regularization_l=regularization_lambda)
                                        net.add(FullyConnectedLayer(input_size, M, sigmoid, sigmoid_prime, init_f))
                                        for i in range(N):  # N -hidden- layers, plus input and output layers
                                            net.add(FullyConnectedLayer(M, M, sigmoid, sigmoid_prime, init_f))
                                        net.add(FullyConnectedLayer(M, output_size, sigmoid, sigmoid_prime, init_f))
                                        if (verbose): net.summary()

                                        if not(X_validation is None):
                                            history, val_history = net.training_loop(X, y, X_validation=X_validation, Y_validation=Y_validation, epochs=epochs, learning_rate=E, batch_size=B, verbose=verbose, early_stopping=early_stopping)
                                        else:
                                            history = net.training_loop(X, y, epochs=epochs, learning_rate=E, batch_size=B, verbose=verbose, early_stopping=early_stopping)

                                        results.append({"loss":history[-1],
                                                        "layers":N,
                                                        "units":M,
                                                        "learning_rate":E,
                                                        "batch_size":B,
                                                        "init_function":init_f,
                                                        "momentum":momentum,
                                                        "regularizator":regularizatorname,
                                                        "regularization_lambda":regularization_lambda})

                                        tested += 1
                                        progress = tested/n_combinations
                                        digits = len(str(n_combinations))
                                        formattedtested = ("{:0"+str(digits)+"d}").format(tested)
                                        elapsedtime = time.time() - start
                                        ETAtime = time.gmtime((elapsedtime * (n_combinations / tested)) - elapsedtime)
                                        ETA = "ETA " + time.strftime("%Hh %Mm %Ss", ETAtime)
                                        update_progress(progress, prefix = ETA + " " + formattedtested + "/" + str(n_combinations), barlength=80)
                                        if (verbose): print("")
    finally:
        results.sort(key = lambda x: x['loss'], reverse=False)
        if (verbose):
            print("Best: " + str(results[0]))
            print("Top 10:")
            for i in range (0, 10):
                print("\t" + str(results[i]))
        return results[0:10]
