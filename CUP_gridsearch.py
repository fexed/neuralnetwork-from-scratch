import numpy as np
from grid_search import grid_search
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import tr_vl_split


X, Y = load_cup(verbose=True, test=False)
xtr, xvl, ytr, yvl = tr_vl_split(X, Y, ratio = 0.25)

res = grid_search(10, 2, xtr, ytr, X_validation=xvl, Y_validation=yvl, layers=list(range(1, 4)), units=list(range(5, 20, 5)), learning_rates=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], batch_sizes=[1], init_functions=["normalized_xavier"], momentums=[0.8], regularizators=[None, L2(l = 0.001), L2(l = 0.005), L2(l = 0.010), L2(l = 0.015)], epochs=1000, verbose=False, early_stopping=150)
result_file = open("datasets/CUP/grid_search/results.txt", "w")
print("CUP\n")
result_file.write("CUP:\n")
for i in range (0, 10):
    result_file.write(str(i+1)+": "+str(res[i]))
    print("\t" + str(res[i]))
result_file.close()
