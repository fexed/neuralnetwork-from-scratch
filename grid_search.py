import numpy as np
from activationfunctions import sigmoid, sigmoid_prime
from losses import binary_crossentropy, binary_crossentropy_prime
from layers import FullyConnectedLayer
from neuralnetwork import Network
from regularizators import L2, weight_decay
from utils import update_progress


def grid_search(input_size, output_size, X, y, X_validation=None, Y_validation=None, layers=list(range(5)), units=list(range(5, 100, 5)), learning_rates=list(np.arange(0.01, 0.1, 0.01)), batch_sizes=None, init_functions=["xavier", "normalized_xavier", "he"], momentums=[0, 0.8, 0.9, 0.99, 0.999], regularizators=[None, "L2", "weight_decay"], epochs=500, verbose=True, early_stopping=25):
    n_combinations = len(layers)*len(units)*len(learning_rates)*len(init_functions)*len(momentums)*len(regularizators)*len(batch_sizes)
    if (verbose): print("Grid search on " + str(n_combinations) + " combinations")

    if (batch_sizes==None):
        batch_sizes=[1, input_size]
    results, parameters = [], []  # to store the results and return the best one

    tested = 0
    for N in layers:
        for M in units:
            for E in learning_rates:
                for momentum in momentums:
                    for regularizatorname in regularizators:
                        if regularizatorname == "L2": regularizator = L2
                        elif regularizatorname == "weight_decay": regularizator = weight_decay
                        else: regularizator = None
                        for init_f in init_functions:
                            for B in batch_sizes:
                                net = Network("GRIDSEARCH_" + str(N) + "L_" + str(M) + "U_" + str(E) + "LR", binary_crossentropy, binary_crossentropy_prime, momentum=momentum, regularizator=regularizator)
                                net.add(FullyConnectedLayer(input_size, M, sigmoid, sigmoid_prime, init_f))
                                for i in range(N):  # N -hidden- layers, plus input and output layers
                                    net.add(FullyConnectedLayer(M, M, sigmoid, sigmoid_prime, init_f))
                                net.add(FullyConnectedLayer(M, output_size, sigmoid, sigmoid_prime, init_f))
                                if (verbose): net.summary()

                                if not(X_validation is None):
                                    history, val_history = net.training_loop(X, y, X_validation=X_validation, Y_validation=Y_validation, epochs=epochs, learning_rate=E, batch_size=B, verbose=verbose, early_stopping=early_stopping)
                                    results.append(history[-1])
                                else:
                                    history = net.training_loop(X, y, epochs=epochs, learning_rate=E, batch_size=B, verbose=verbose, early_stopping=early_stopping)
                                    results.append(history[-1])

                                parameters.append({"layers":N, "units":M, "learning_rate":E, "batch_size":B, "init_function":init_f, "momentum":momentum, "regularizator":regularizatorname})
                                tested += 1
                                progress = tested/n_combinations
                                digits = len(str(n_combinations))
                                formattedtested = ("{:0"+str(digits)+"d}").format(tested)
                                update_progress(progress, prefix = formattedtested + "/" + str(n_combinations))
                                if (verbose): print("")

    results, parameters = zip(*sorted(zip(results, parameters)))  # sort both lists
    if (verbose):
        print("Best: " + "{:.5f}".format(results[0]) + " with " + str(parameters[0]))
        print("Top 10:")
        for i in range (0, 10):
            print("\t" + "{:.5f}".format(results[i]) + " with " + str(parameters[i]))
    return parameters[0:10], results[0:10]
