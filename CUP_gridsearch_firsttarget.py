import numpy as np
from grid_search import grid_search
from regularizators import L1, L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import tr_vl_split


X, Y = load_cup(verbose=True, test=False)
Y = Y[0:,0:,1]  # second target
xtr, xvl, ytr, yvl =  tr_vl_split(X, Y, ratio=0.2)

res = grid_search(10, 1, xtr, ytr, X_validation=xvl, Y_validation=yvl, layers=[1, 2, 3], units=list(range(20, 35)), learning_rates=[0.001], batch_sizes=[1], init_functions=["normalized_xavier"], momentums=[0], regularizators=[None], epochs=1000, verbose=False, early_stopping=25)

result_file = open("datasets/CUP/grid_search/results_featurestosecondtarget.txt", "w")
result_file.write("CUP:\n")
try:
    for i in range (0, 10):
        result_file.write(str(i+1)+": "+str(res[i]))
        print("\t" + str(res[i]))
    result_file.close()
except IndexError:
    pass

# best {'loss': 0.4360415384225904, 'layers': 2, 'units': 30, 'learning_rate': 0.001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0, 'regularizator': None, 'regularization_lambda': 0, 'dropout': 0, 'epochs': 291}


"""
X, Y = load_cup(verbose=True, test=False, forfirst=True)
xtr, xvl, ytr, yvl =  tr_vl_split(X, Y, ratio=0.2)

try:
    res = grid_search(11, 1, xtr, ytr, X_validation=xvl, Y_validation=yvl, layers=[1], units=[28,32], learning_rates=[0.1, 0.001, 0.0001], batch_sizes=[1], init_functions=["normalized_xavier"], momentums=[0, 0.5, 0.8, 0.99], regularizators=[None, L1(l = 0.005), L2(l = 0.005)], epochs=1000, verbose=False, early_stopping=15)
    result_file = open("datasets/CUP/grid_search/results_featuresandsecondtofirsttarget.txt", "w")
    result_file.write("CUP:\n")
    for i in range (0, 10):
        result_file.write(str(i+1)+": "+str(res[i]))
        print("\t" + str(res[i]))
    result_file.close()
except IndexError:
    pass

# best {'loss': 0.3522742811185453, 'layers': 1, 'units': 28, 'learning_rate': 0.001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0, 'regularizator': 'None', 'regularization_lambda': 0, 'dropout': 0, 'epochs': 97}

{'loss': 0.3411194331370274, 'layers': 1, 'units': 28, 'learning_rate': 0.0001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.8, 'regularizator': 'None', 'regularization_lambda': 0, 'dropout': 0, 'epochs': 184}
        {'loss': 0.34376532071935534, 'layers': 1, 'units': 32, 'learning_rate': 0.0001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.8, 'regularizator': 'None', 'regularization_lambda': 0, 'dropout': 0, 'epochs': 122}
        {'loss': 0.3471768979515812, 'layers': 1, 'units': 28, 'learning_rate': 0.001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0.5, 'regularizator': 'None', 'regularization_lambda': 0, 'dropout': 0, 'epochs': 64}
        {'loss': 0.3591069099279923, 'layers': 1, 'units': 32, 'learning_rate': 0.001, 'batch_size': 1, 'init_function': 'normalized_xavier', 'momentum': 0, 'regularizator': 'None', 'regularization_lambda': 0, 'dropout': 0, 'epochs': 82}
"""
