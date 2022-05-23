import numpy as np
from grid_search import grid_search
from regularizators import L2
from dataset_loader import load_cup
from preprocessing import continuous_standardizer, min_max_normalizer
from utils import tr_vl_split


X, Y = load_cup(verbose=True, file="training")

res = grid_search(10, 2, X, Y, layers=[0], units=list(range(20, 26, 1)), learning_rates=[0.0025, 0.00125, 0.001, 0.000625], batch_sizes=[1], init_functions=["normalized_xavier"], momentums=[0], regularizators=[L2(l = 1e-4), L2(l = 1e-5), L2(l = 1e-6)], dropouts=[0, 0.5], nesterovs=[False], epochs=2000, verbose=False, early_stopping=150)
result_file = open("datasets/CUP/grid_search/results.txt", "w")
print("CUP\n")
result_file.write("CUP:\n")
for i in range (0, 10):
    result_file.write(str(i+1)+": "+str(res[i]))
    print("\t" + str(res[i]))
result_file.close()
